from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import torch

from models.replay.EmbeddingReplayStore import EmbeddingReplayStore
from models.replay.utils import upsample_per_class
from models.samplers.MemoryExpansionBuffer import MemoryExpansionBuffer


@runtime_checkable
class ReplayStore(Protocol):
    def add_task_samples(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None: ...

    def get_memory_samples(
        self,
        classes: List[int],
        buffer_state: Optional[Dict[int, torch.Tensor]] = None,
        samples_per_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def maintain(self) -> None: ...

    def get_buffer_state(self) -> Dict[int, torch.Tensor]: ...

    def get_class_distribution(
        self, buffer: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[int, int]: ...

    def __len__(self) -> int: ...


class BufferReplayAdapter:
    """Wraps MemoryExpansionBuffer with the ReplayStore maintain() API."""

    def __init__(self, buffer: MemoryExpansionBuffer):
        self._buffer = buffer

    def add_task_samples(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        self._buffer.add_task_samples(embeddings, labels)

    def get_memory_samples(
        self,
        classes: List[int],
        buffer_state: Optional[Dict[int, torch.Tensor]] = None,
        samples_per_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self._buffer.get_memory_samples(classes, buffer_state=buffer_state)
        if samples_per_class is not None and x.numel() > 0:
            x, y = upsample_per_class(x, y, samples_per_class)
        return x, y

    def maintain(self) -> None:
        self._buffer._maintain_buffer()

    def get_buffer_state(self) -> Dict[int, torch.Tensor]:
        return self._buffer.get_buffer_state()

    def get_class_distribution(
        self, buffer: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[int, int]:
        return self._buffer.get_class_distribution(buffer=buffer)

    def __len__(self) -> int:
        return len(self._buffer)


def _make_embedding_store(
    inc: Dict[str, Any],
    model_cfg: Dict[str, Any],
    device: str,
    anchor_per_class: int,
    use_real_buffer: bool,
    use_teacher_kl: bool,
    gen_replay_ratio: float,
) -> EmbeddingReplayStore:
    """
    Shared factory for all CVAE-backed replay stores.  Parameters that differ
    between ``cvae_hybrid``, ``cvae_full``, and ``cvae_pure`` are supplied by
    the caller.
    """
    normalize_embeddings = bool(model_cfg.get("normalize", True))
    return EmbeddingReplayStore(
        embed_dim=int(inc.get("cvae_embed_dim", 768)),
        memory_size=int(inc.get("buffer_size", 2000)),
        anchor_per_class=anchor_per_class,
        generated_per_class=int(inc.get("generated_per_class", 15)),
        latent_dim=int(inc.get("cvae_latent_dim", 64)),
        class_embed_dim=int(inc.get("cvae_class_embed_dim", 64)),
        hidden_dim=int(inc.get("cvae_hidden_dim", 256)),
        train_steps=int(
            inc.get(
                "cvae_train_max_steps",
                inc.get("cvae_train_steps", 2000),
            )
        ),
        train_min_steps=int(inc.get("cvae_train_min_steps", 50)),
        train_eval_every=int(inc.get("cvae_train_eval_every", 20)),
        train_patience=int(inc.get("cvae_train_patience", 3)),
        recon_cosine_stop=float(inc.get("cvae_recon_cosine_stop", 0.93)),
        gen_cosine_stop=float(inc.get("cvae_gen_cosine_stop", 0.88)),
        gen_eval_samples_per_class=int(
            inc.get("cvae_gen_eval_samples_per_class", 32)
        ),
        train_on=str(inc.get("cvae_train_on", "real_buffer")),
        beta=float(inc.get("cvae_beta", 0.05)),
        gen_replay_ratio=gen_replay_ratio,
        use_teacher_kl=use_teacher_kl,
        teacher_kl_weight=float(inc.get("cvae_teacher_kl_weight", 0.1)),
        device=device,
        max_classes=int(inc.get("max_classes", 1000)),
        stats_loss_weight=float(inc.get("cvae_stats_loss_weight", 0.01)),
        feature_matching_weight=float(inc.get("cvae_feature_matching_weight", 0.1)),
        replay_debug=bool(inc.get("replay_debug", False)),
        normalize_embeddings=normalize_embeddings,
        kl_anneal_steps=int(inc.get("cvae_kl_anneal_steps", 200)),
        beta_floor=float(inc.get("cvae_beta_floor", 1e-4)),
        use_real_buffer=use_real_buffer,
        ref_buffer_per_class=int(inc.get("ref_buffer_per_class", 0)),
        min_sampling_std=float(inc.get("cvae_min_sampling_std", 0.2)),
        ref_buffer_anchor_cvae=bool(inc.get("ref_buffer_anchor_cvae", False)),
    )


def create_replay_store(
    config: Dict[str, Any],
    sampling_strategy: Any,
    device: str = "cuda",
) -> ReplayStore:
    """
    Factory for replay backends.

    Backends
    --------
    ``buffer``
        Classic exemplar buffer (``MemoryExpansionBuffer``) wrapped in
        ``BufferReplayAdapter``.  No generative model is used.

    ``cvae_hybrid``
        Hybrid store: real exemplar buffer + CVAE.  Replay prefers stored reals
        and uses the CVAE only to fill up to the requested per-class count.
        ``anchor_per_class`` controls the number of herding anchors kept per
        class (default 5).

    ``cvae_full``
        Like ``cvae_hybrid`` but with ``anchor_per_class=0``.  The real buffer
        is still maintained; the CVAE is the preferred source for replay when
        the quality gate is met.  Teacher KL distillation is enabled by default.

    ``cvae_pure``  ← **new, recommended**
        Fully buffer-free generative replay.  No real embeddings are stored for
        classifier replay.  The CVAE is trained on current-task real embeddings
        plus synthetically generated old-class embeddings from the previous CVAE
        snapshot, balanced to the same per-class count.  ``get_memory_samples()``
        always draws from the CVAE with no fallback.

        Configuration knobs specific to ``cvae_pure``:
          - ``cvae_kl_anneal_steps``  (default 200) — β warmup length.
          - ``cvae_beta_floor``       (default 1e-4) — initial β.
          - ``cvae_hidden_dim``       (default 256)  — MLP hidden size.
          - ``cvae_feature_matching_weight`` (default 0.1) — feature matching.
          - ``cvae_teacher_kl``       (default True) — teacher KL distillation.
    """
    inc = config.get("incremental", config)
    model_cfg = config.get("model", {})
    backend = inc.get("replay_backend", "buffer").lower()

    if backend == "buffer":
        buffer = MemoryExpansionBuffer(
            total_memory_size=inc["buffer_size"],
            sampling_strategy=sampling_strategy,
        )
        return BufferReplayAdapter(buffer)

    if backend in ("cvae_hybrid", "cvae_full"):
        anchor_per_class = int(inc.get("anchor_per_class", 5))
        if backend == "cvae_full":
            anchor_per_class = 0
        store = _make_embedding_store(
            inc=inc,
            model_cfg=model_cfg,
            device=device,
            anchor_per_class=anchor_per_class,
            use_real_buffer=True,
            use_teacher_kl=bool(
                inc.get("cvae_teacher_kl", backend == "cvae_full")
            ),
            gen_replay_ratio=float(inc.get("gen_replay_ratio", 0.3)),
        )
        store.replay_generate_from_cvae = bool(
            inc.get("replay_generate_from_cvae", True)
        )
        return store

    if backend == "cvae_pure":
        store = _make_embedding_store(
            inc=inc,
            model_cfg=model_cfg,
            device=device,
            anchor_per_class=0,
            use_real_buffer=False,
            # Teacher KL distillation is always enabled in pure mode:
            # it is one of the two mechanisms that prevent catastrophic
            # forgetting inside the CVAE itself.
            use_teacher_kl=bool(inc.get("cvae_teacher_kl", True)),
            # gen_replay_ratio is 0.0 because maintain() pre-generates
            # old-class data before calling fit(), making internal mixing
            # unnecessary.
            gen_replay_ratio=0.0,
        )
        store.replay_generate_from_cvae = True
        return store

    raise ValueError(
        f"Unknown replay_backend: '{backend}'. "
        "Choose from: buffer, cvae_hybrid, cvae_full, cvae_pure."
    )
