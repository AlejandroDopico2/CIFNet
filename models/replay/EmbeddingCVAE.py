from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingCVAE(nn.Module):
    """
    Class-conditional VAE operating entirely in backbone-embedding space.

    Design changes vs. the previous version
    ----------------------------------------
    1.  **Class-conditional encoder** — the encoder now receives ``[x ‖ class_embed(y)]``
        as input, realising a proper CVAE posterior  q(z | x, y).  The previous
        encoder was unconditional (q(z | x) only), which meant class information
        was discarded in the latent space and the model was really just a
        conditional decoder.  This fix is critical for per-class generation quality.

    2.  **Deeper architecture with LayerNorm + GELU** — 3-layer MLPs replace
        2-layer ReLU nets.  LayerNorm prevents activation collapse on the
        unit-sphere feature manifold; GELU provides smoother gradients than ReLU
        for regression-style embedding reconstruction.  Default hidden_dim is
        raised from 128 → 256 and latent_dim from 32 → 64.

    3.  **KL annealing** — β is linearly warmed up from ``beta_floor`` to the
        caller-supplied ``beta`` over ``kl_anneal_steps`` gradient steps.  This
        prevents posterior collapse during early training where the reconstruction
        gradient is small relative to the KL penalty.

    4.  **Per-class latent priors (non-parametric)** — per-class latent statistics
        (μ_c, log σ²_c) are estimated via an exponential moving average over the
        encoder posterior during training.  At generation time, samples are drawn
        from  N(μ_c, σ_c)  rather than the vanilla  N(0, I),  which keeps generated
        embeddings on the learned per-class manifold and substantially improves
        generation quality for classes already seen.

    5.  **Feature-matching loss** — per-class mean and variance of reconstructed
        embeddings are penalised against the corresponding statistics of real
        embeddings.  This enforces distributional alignment beyond reconstruction
        quality.

    6.  **Gradient clipping** — max-norm clipping (1.0) is applied inside ``fit()``
        to stabilise training on the potentially high-variance embedding space.

    Teacher KL distillation (from the previous version) is retained unchanged:
    it penalises the student posterior from drifting away from the teacher's
    posterior on the same inputs, which mitigates catastrophic forgetting inside
    the generative model itself.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        latent_dim: int = 64,
        class_embed_dim: int = 64,
        hidden_dim: int = 256,
        normalize_embeddings: bool = True,
        kl_anneal_steps: int = 200,
        beta_floor: float = 1e-4,
        min_sampling_std: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.normalize_embeddings = normalize_embeddings
        self.kl_anneal_steps = kl_anneal_steps
        self.beta_floor = beta_floor
        self._hidden_dim = hidden_dim
        # Hard floor on per-class sampling std.  Prevents complete collapse to a
        # point when _class_latent_logvar carries a degenerate small value.
        # Set to 0.0 to disable.
        self._min_sampling_std = min_sampling_std

        # Shared class embedding used by both encoder and decoder.
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)

        # Encoder: class-conditional posterior  q(z | x, y).
        # Input: [x ‖ class_embed(y)]  →  hidden  →  μ, log σ²
        enc_in = embed_dim + class_embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: p(x | z, y).
        # Input: [z ‖ class_embed(y)]  →  hidden  →  embedding
        dec_in = latent_dim + class_embed_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Non-parametric per-class latent statistics estimated via EMA.
        # Populated during training; used for class-conditional sampling.
        self._class_latent_mu: Dict[int, torch.Tensor] = {}
        self._class_latent_logvar: Dict[int, torch.Tensor] = {}
        self._class_prior_initialized: Set[int] = set()

        # Global gradient-step counter used for KL annealing.
        self._train_step: int = 0

    # ------------------------------------------------------------------
    # Class-embedding expansion (called when new classes are encountered)
    # ------------------------------------------------------------------

    def _expand_classes(self, new_num_classes: int) -> None:
        if new_num_classes <= self.num_classes:
            return
        old_embed = self.class_embed
        new_embed = nn.Embedding(new_num_classes, old_embed.embedding_dim).to(
            old_embed.weight.device
        )
        with torch.no_grad():
            new_embed.weight[: old_embed.num_embeddings] = old_embed.weight
        self.class_embed = new_embed
        self.num_classes = new_num_classes
        # _class_latent_mu / _class_latent_logvar are plain dicts;
        # they grow automatically as new classes are encountered.

    # ------------------------------------------------------------------
    # Core VAE components
    # ------------------------------------------------------------------

    def encode(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Class-conditional posterior  q(z | x, y)."""
        if y.max().item() >= self.num_classes:
            self._expand_classes(int(y.max().item()) + 1)
        cy = self.class_embed(y)
        h = self.encoder(torch.cat([x, cy], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.max().item() >= self.num_classes:
            self._expand_classes(int(y.max().item()) + 1)
        cy = self.class_embed(y)
        out = self.decoder(torch.cat([z, cy], dim=-1))
        if self.normalize_embeddings:
            return F.normalize(out, p=2, dim=-1)
        return out

    def reconstruct(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction via the posterior mean μ."""
        mu, _ = self.encode(x, y)
        return self.decode(mu, y)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar, z

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x, p=2, dim=-1) * F.normalize(y, p=2, dim=-1)).sum(dim=-1)

    @staticmethod
    def cosine_recon_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return (1.0 - EmbeddingCVAE.cosine_similarity(x, recon)).mean()

    @staticmethod
    def mse_recon_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, x)

    def recon_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        if self.normalize_embeddings:
            return self.cosine_recon_loss(x, recon)
        return self.mse_recon_loss(x, recon)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    @staticmethod
    def kl_distill(
        mu_s: torch.Tensor,
        logvar_s: torch.Tensor,
        mu_t: torch.Tensor,
        logvar_t: torch.Tensor,
    ) -> torch.Tensor:
        """KL(student posterior ‖ teacher posterior) — prevents CVAE forgetting."""
        var_s = logvar_s.exp()
        var_t = logvar_t.exp()
        return 0.5 * (
            logvar_t - logvar_s
            + (var_s + (mu_s - mu_t).pow(2)) / var_t.clamp(min=1e-8)
            - 1
        ).sum(dim=-1).mean()

    @staticmethod
    def feature_matching_loss(
        recon: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-class first- and second-moment matching between reconstructed and
        real embeddings.  Encourages the generator to preserve class geometry.
        Requires ≥ 2 samples per class to compute variance; single-sample
        classes contribute only the mean term.
        """
        total = torch.tensor(0.0, device=recon.device)
        n_terms = 0
        for c in y.unique().tolist():
            mask = y == int(c)
            n_c = int(mask.sum().item())
            if n_c == 0:
                continue
            r = recon[mask]
            t = x[mask].detach()
            total = total + F.mse_loss(r.mean(dim=0), t.mean(dim=0))
            n_terms += 1
            if n_c >= 2:
                total = total + F.mse_loss(r.var(dim=0), t.var(dim=0))
                n_terms += 1
        return total / max(n_terms, 1)

    # ------------------------------------------------------------------
    # KL annealing
    # ------------------------------------------------------------------

    def _anneal_beta(self, beta: float) -> float:
        """Linear warmup: β grows from beta_floor → beta over kl_anneal_steps."""
        if self.kl_anneal_steps <= 0:
            return beta
        progress = min(1.0, self._train_step / max(self.kl_anneal_steps, 1))
        return self.beta_floor + progress * (beta - self.beta_floor)

    # ------------------------------------------------------------------
    # Per-class latent statistics (for class-conditional sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_class_priors(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        y: torch.Tensor,
        momentum: float = 0.9,
    ) -> None:
        """
        EMA update of per-class latent statistics using the **law of total variance**.

        Previous implementation stored ``E[logvar_i]`` (average within-sample
        posterior uncertainty) as the class sampling spread.  This was incorrect:
        it completely ignored the **between-sample** diversity — how spread out
        the posterior *means* are across different data points of the same class.
        With even moderate KL regularisation the individual posteriors become
        tight (small ``sigma_i``), making ``E[logvar_i]`` collapse towards ``-∞``
        and generating near-constant embeddings regardless of class size.

        Correct decomposition (law of total variance):
        ::
            Var[z for class c]
              = E[Var[z | x_i]]          (within-sample: avg individual posterior variance)
              + Var[E[z | x_i]]          (between-sample: variance of posterior *means*)

        The between-sample term reflects how much diversity the class actually
        has in latent space and is independent of KL regularisation strength.
        """
        device = mu.device
        for c in y.unique().tolist():
            c = int(c)
            mask = y == c
            n_c = int(mask.sum().item())
            if n_c == 0:
                continue

            mu_batch = mu[mask].detach()     # (n_c, latent_dim)
            lv_batch = logvar[mask].detach() # (n_c, latent_dim)

            mu_c = mu_batch.mean(dim=0)

            # Within-sample variance: E[sigma_i²]
            within_var = torch.exp(lv_batch).mean(dim=0)

            # Between-sample variance: Var[mu_i across different x_i of class c]
            # This is the dominant source of diversity when KL is non-negligible.
            between_var = (
                mu_batch.var(dim=0, unbiased=True)
                if n_c > 1
                else torch.zeros_like(mu_c)
            )

            total_var = (within_var + between_var).clamp(min=1e-4)
            lv_c_total = torch.log(total_var)

            if c not in self._class_prior_initialized:
                self._class_latent_mu[c] = mu_c.cpu()
                self._class_latent_logvar[c] = lv_c_total.cpu()
                self._class_prior_initialized.add(c)
            else:
                prev_mu = self._class_latent_mu[c].to(device)
                prev_lv = self._class_latent_logvar[c].to(device)
                self._class_latent_mu[c] = (
                    momentum * prev_mu + (1.0 - momentum) * mu_c
                ).cpu()
                self._class_latent_logvar[c] = (
                    momentum * prev_lv + (1.0 - momentum) * lv_c_total
                ).cpu()

    # ------------------------------------------------------------------
    # Full loss
    # ------------------------------------------------------------------

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        beta: float = 0.05,
        teacher: Optional["EmbeddingCVAE"] = None,
        teacher_kl_weight: float = 0.1,
        stats_loss_weight: float = 0.0,
        class_means: Optional[Dict[int, torch.Tensor]] = None,
        feature_matching_weight: float = 0.1,
    ) -> torch.Tensor:
        recon, mu, logvar, _ = self.forward(x, y)

        effective_beta = self._anneal_beta(beta)
        total = self.recon_loss(x, recon) + effective_beta * self.kl_loss(mu, logvar)

        # Feature matching: explicitly align per-class moments.
        if feature_matching_weight > 0:
            total = total + feature_matching_weight * self.feature_matching_loss(
                recon, x, y
            )

        # Prototype mean-matching (legacy compatibility with stats_loss_weight config).
        if stats_loss_weight > 0 and class_means:
            batch_means, targets = [], []
            for c in y.unique().tolist():
                ci = int(c)
                mask = y == c
                if mask.any() and ci in class_means:
                    batch_means.append(recon[mask].mean(dim=0))
                    targets.append(class_means[ci].to(recon.device))
            if batch_means:
                bm = torch.stack(batch_means)
                tm = torch.stack(targets)
                total = total + stats_loss_weight * F.mse_loss(bm, tm)

        # Teacher KL distillation: prevent catastrophic forgetting inside the CVAE.
        # The student posterior is penalised for diverging from the teacher's
        # posterior on the same (x, y) pairs.
        if teacher is not None and teacher_kl_weight > 0:
            with torch.no_grad():
                mu_t, logvar_t = teacher.encode(x, y)
            total = total + teacher_kl_weight * self.kl_distill(
                mu, logvar, mu_t, logvar_t
            )

        return total

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_reconstruction_cosine(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> float:
        """Recon quality: cosine (normalised mode) or 1 − relative-MSE (raw mode)."""
        if x.numel() == 0:
            return 0.0
        self.eval()
        recon = self.reconstruct(x, y)
        if self.normalize_embeddings:
            return float(self.cosine_similarity(x, recon).mean().item())
        mse = F.mse_loss(recon, x, reduction="none").mean(dim=-1)
        scale = x.pow(2).mean(dim=-1).clamp(min=1e-8)
        rel = (mse / scale).clamp(max=1.0)
        return float((1.0 - rel).mean().item())

    @torch.no_grad()
    def eval_generation_vs_reals(
        self,
        real_by_class: Dict[int, torch.Tensor],
        samples_per_class: int = 32,
        device: Optional[torch.device] = None,
    ) -> float:
        """
        Mean max-cosine of prior samples to stored real exemplars per class.
        Measures whether generated replay lies on the real embedding manifold.
        """
        if not real_by_class:
            return 0.0
        self.eval()
        device = device or next(self.parameters()).device
        scores: List[float] = []
        for cls, reals in real_by_class.items():
            if cls >= self.num_classes or reals.numel() == 0:
                continue
            reals = reals.float().to(device)
            n = min(samples_per_class, 128)
            gen, _ = self.sample([int(cls)], n, device)
            if gen.numel() == 0:
                continue
            cos = F.cosine_similarity(
                gen.unsqueeze(1), reals.unsqueeze(0), dim=-1
            )
            scores.append(float(cos.max(dim=1).values.mean().item()))
        return float(sum(scores) / len(scores)) if scores else 0.0

    # ------------------------------------------------------------------
    # Sampling from the class-conditional prior
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        class_ids: List[int],
        n_per_class: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw *n_per_class* samples for each class in *class_ids*.

        When per-class latent statistics have been estimated (i.e. the model
        has been trained on at least one sample from class c), samples are
        drawn from the learned class-conditional prior  N(μ_c, σ_c).
        Otherwise the standard Gaussian  N(0, I)  is used as a fallback.

        This significantly improves generation quality compared with always
        sampling from  N(0, I),  because it places the latent codes in the
        region of latent space that the decoder has learned to decode for
        each class.
        """
        if n_per_class <= 0 or not class_ids:
            return (
                torch.empty(0, self.embed_dim, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        embeddings: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []

        for c in class_ids:
            if c >= self.num_classes:
                continue
            y = torch.full((n_per_class,), c, dtype=torch.long, device=device)
            if c in self._class_latent_mu:
                mu_c = self._class_latent_mu[c].to(device)
                logvar_c = self._class_latent_logvar[c].to(device)
                # _class_latent_logvar now stores log(total_var) = log(between +
                # within), so std_c reflects the true class spread in latent
                # space.  The min floor prevents complete point-collapse in edge
                # cases (e.g. single-sample classes or very early tasks).
                std_c = torch.exp(0.5 * logvar_c).clamp(min=self._min_sampling_std)
                z = (
                    mu_c.unsqueeze(0)
                    + torch.randn(n_per_class, self.latent_dim, device=device)
                    * std_c.unsqueeze(0)
                )
            else:
                z = torch.randn(n_per_class, self.latent_dim, device=device)
            embeddings.append(self.decode(z, y))
            labels.append(y)

        if not embeddings:
            return (
                torch.empty(0, self.embed_dim, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        beta: float = 0.05,
        lr: float = 1e-3,
        gen_replay_ratio: float = 0.0,
        teacher: Optional["EmbeddingCVAE"] = None,
        teacher_kl_weight: float = 0.1,
        class_means: Optional[Dict[int, torch.Tensor]] = None,
        stats_loss_weight: float = 0.01,
        feature_matching_weight: float = 0.1,
        current_task_classes: Optional[List[int]] = None,
        real_by_class: Optional[Dict[int, torch.Tensor]] = None,
        # Adaptive stopping (steps is legacy alias for max_steps)
        steps: Optional[int] = None,
        max_steps: int = 2000,
        min_steps: int = 50,
        eval_every: int = 20,
        patience: int = 3,
        recon_cosine_stop: float = 0.93,
        gen_cosine_stop: float = 0.0,
        gen_eval_samples_per_class: int = 32,
    ) -> Dict[str, Any]:
        """
        Train until reconstruction / generation metrics satisfy thresholds, or
        until max_steps.

        Adaptive stopping fires after *patience* consecutive evaluations where
        recon_cosine ≥ recon_cosine_stop (and gen_cosine ≥ gen_cosine_stop if
        gen_cosine_stop > 0).

        Per-class latent statistics are updated every eval_every // 2 steps so
        that the class-conditional sampling prior stays current throughout
        training.

        Gradient clipping (max-norm 1.0) is applied at every step to stabilise
        training on potentially high-variance embedding spaces.
        """
        if x.numel() == 0:
            return {"steps": 0, "stopped": "empty_data"}

        if steps is not None:
            max_steps = steps

        device = x.device
        max_class = int(y.max().item()) + 1
        if max_class > self.num_classes:
            self._expand_classes(max_class)

        # Evaluation reference: real embeddings per class.
        real_eval: Dict[int, torch.Tensor] = {}
        if real_by_class is not None:
            real_eval = real_by_class
        if not real_eval:
            for c in y.unique().tolist():
                ci = int(c)
                mask = y == ci
                if mask.any():
                    real_eval[ci] = x[mask].detach()

        # Train / validation split.
        n = x.size(0)
        perm = torch.randperm(n, device=device)
        n_val = max(1, min(n // 5, 512))
        x_val, y_val = x[perm[:n_val]], y[perm[:n_val]]
        x_train, y_train = x[perm[n_val:]], y[perm[n_val:]]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        batch_size = min(256, x_train.size(0))
        step = 0
        stable = 0
        last_metrics: Dict[str, float] = {}
        use_gen_stop = gen_cosine_stop > 0.0 and bool(real_eval)
        prior_update_every = max(eval_every // 2, 1)

        while step < max_steps:
            perm_t = torch.randperm(x_train.size(0), device=device)
            bx = x_train[perm_t[:batch_size]]
            by = y_train[perm_t[:batch_size]]

            # Legacy internal generative replay.  When maintain() pre-generates
            # old-class data before calling fit(), gen_replay_ratio=0.0 and this
            # block is skipped.
            if (
                gen_replay_ratio > 0
                and teacher is not None
                and teacher.num_classes > 0
            ):
                skip = set(current_task_classes or [])
                old_classes = [c for c in range(teacher.num_classes) if c not in skip]
                if old_classes:
                    n_gen = max(1, int(batch_size * gen_replay_ratio))
                    n_per = max(1, n_gen // len(old_classes))
                    gx, gy = teacher.sample(old_classes[:32], n_per, device)
                    if gx.numel() > 0:
                        take = min(gx.size(0), n_gen)
                        bx = torch.cat([bx, gx[:take]], dim=0)
                        by = torch.cat([by, gy[:take]], dim=0)

            optimizer.zero_grad()
            loss_val = self.loss(
                bx,
                by,
                beta=beta,
                teacher=teacher,
                teacher_kl_weight=teacher_kl_weight,
                stats_loss_weight=stats_loss_weight,
                feature_matching_weight=feature_matching_weight,
                class_means=class_means,
            )
            loss_val.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            self._train_step += 1
            step += 1

            # Periodically update per-class latent statistics.
            if step % prior_update_every == 0:
                with torch.no_grad():
                    mu_b, lv_b = self.encode(bx.detach(), by.detach())
                self._update_class_priors(mu_b, lv_b, by.detach())

            if step < min_steps or step % eval_every != 0:
                continue

            recon_cos = self.eval_reconstruction_cosine(x_val, y_val)
            gen_cos = (
                self.eval_generation_vs_reals(
                    real_eval,
                    samples_per_class=gen_eval_samples_per_class,
                    device=device,
                )
                if use_gen_stop
                else 1.0
            )
            last_metrics = {
                "recon_cosine": recon_cos,
                "gen_proto_cosine": gen_cos,
                "train_loss": float(loss_val.item()),
                "effective_beta": self._anneal_beta(beta),
            }
            self.train()

            recon_ok = recon_cos >= recon_cosine_stop
            gen_ok = (not use_gen_stop) or (gen_cos >= gen_cosine_stop)
            if recon_ok and gen_ok:
                stable += 1
                if stable >= patience:
                    self._finalise_priors(x, y)
                    self.eval()
                    return {
                        "steps": step,
                        "stopped": "quality_threshold",
                        **last_metrics,
                    }
            else:
                stable = 0

        self._finalise_priors(x, y)
        self.eval()

        if not last_metrics:
            last_metrics = {
                "recon_cosine": self.eval_reconstruction_cosine(x_val, y_val),
                "gen_proto_cosine": (
                    self.eval_generation_vs_reals(
                        real_eval,
                        samples_per_class=gen_eval_samples_per_class,
                        device=device,
                    )
                    if use_gen_stop
                    else 0.0
                ),
                "effective_beta": self._anneal_beta(beta),
            }

        return {"steps": step, "stopped": "max_steps", **last_metrics}

    @torch.no_grad()
    def _finalise_priors(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Compute exact per-class latent statistics from the full training set.

        Unlike the EMA updates done during training (which use mini-batch
        estimates with momentum smoothing), this pass has access to *all*
        training data for each class and therefore gives the most accurate
        estimate of the between-sample spread ``Var[mu_i]``.  The result
        *replaces* (rather than averages with) the running EMA estimates,
        making the final sampling distribution as faithful as possible.

        Processing is done class-by-class to stay within GPU memory.
        """
        self.eval()
        for c in y.unique().tolist():
            ci = int(c)
            mask = y == ci
            if not mask.any():
                continue

            x_c = x[mask]
            y_c = torch.full((x_c.size(0),), ci, dtype=torch.long, device=x.device)

            mu_all, lv_all = self.encode(x_c, y_c)

            mu_mean = mu_all.mean(dim=0)

            within_var = torch.exp(lv_all).mean(dim=0)
            between_var = (
                mu_all.var(dim=0, unbiased=True)
                if mu_all.size(0) > 1
                else torch.zeros_like(mu_mean)
            )
            total_var = (within_var + between_var).clamp(min=1e-4)

            # Override EMA with the exact estimate.
            self._class_latent_mu[ci] = mu_mean.detach().cpu()
            self._class_latent_logvar[ci] = torch.log(total_var).detach().cpu()
            self._class_prior_initialized.add(ci)

    # ------------------------------------------------------------------
    # Snapshot (teacher copy for continual training)
    # ------------------------------------------------------------------

    def snapshot(self) -> "EmbeddingCVAE":
        """
        Return a frozen deep copy of this CVAE to use as a teacher for the
        next incremental task.  Copies the per-class latent statistics so that
        the teacher can generate high-quality samples from the class-conditional
        prior.
        """
        clone = EmbeddingCVAE(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            latent_dim=self.latent_dim,
            class_embed_dim=self.class_embed.embedding_dim,
            hidden_dim=self._hidden_dim,
            normalize_embeddings=self.normalize_embeddings,
            kl_anneal_steps=self.kl_anneal_steps,
            beta_floor=self.beta_floor,
            min_sampling_std=self._min_sampling_std,
        )
        clone.load_state_dict(deepcopy(self.state_dict()))
        clone._class_latent_mu = {k: v.clone() for k, v in self._class_latent_mu.items()}
        clone._class_latent_logvar = {
            k: v.clone() for k, v in self._class_latent_logvar.items()
        }
        clone._class_prior_initialized = set(self._class_prior_initialized)
        clone._train_step = self._train_step
        clone.eval()
        return clone.to(next(self.parameters()).device)
