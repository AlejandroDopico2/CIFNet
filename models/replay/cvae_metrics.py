"""Metrics to compare CVAE replay quality against real / buffer embeddings."""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.replay.EmbeddingCVAE import EmbeddingCVAE


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1)


@torch.no_grad()
def class_prototypes(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> Dict[int, torch.Tensor]:
    labels = labels.view(-1).long()
    protos = {}
    for c in labels.unique().tolist():
        mask = labels == c
        protos[int(c)] = _normalize(embeddings[mask]).mean(dim=0)
    return protos


@torch.no_grad()
def prototype_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: Dict[int, torch.Tensor],
) -> float:
    if embeddings.numel() == 0:
        return 0.0
    z = _normalize(embeddings)
    classes = sorted(prototypes.keys())
    P = torch.stack([prototypes[c] for c in classes], dim=0)
    sims = z @ P.T
    preds = torch.tensor([classes[i] for i in sims.argmax(dim=1).tolist()])
    true = labels.view(-1).long().cpu()
    return (preds == true).float().mean().item()


@torch.no_grad()
def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _normalize(a) @ _normalize(b).T


@torch.no_grad()
def mean_cosine_to_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: Dict[int, torch.Tensor],
) -> Tuple[float, Dict[int, float]]:
    z = _normalize(embeddings)
    per_class = {}
    for c in labels.unique().tolist():
        c = int(c)
        if c not in prototypes:
            continue
        mask = labels.view(-1) == c
        sim = (z[mask] * prototypes[c].unsqueeze(0)).sum(dim=-1)
        per_class[c] = sim.mean().item()
    if not per_class:
        return 0.0, per_class
    return sum(per_class.values()) / len(per_class), per_class


@torch.no_grad()
def intra_inter_cosine(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    """Within-class vs between-class mean cosine (higher ratio = better separation)."""
    z = _normalize(embeddings)
    labels = labels.view(-1).long()
    classes = labels.unique().tolist()
    if len(classes) < 2:
        return {"intra_cosine": 1.0, "inter_cosine": 0.0, "intra_over_inter": 0.0}

    intra, inter = [], []
    for i, ci in enumerate(classes):
        mask_i = labels == ci
        zi = z[mask_i]
        if zi.size(0) > 1:
            sim = zi @ zi.T
            intra.extend(sim[torch.triu(torch.ones_like(sim), diagonal=1) == 1].tolist())
        for cj in classes[i + 1 :]:
            mask_j = labels == cj
            inter.extend((zi @ z[mask_j].T).reshape(-1).tolist())

    intra_m = sum(intra) / max(len(intra), 1)
    inter_m = sum(inter) / max(len(inter), 1)
    return {
        "intra_cosine": intra_m,
        "inter_cosine": inter_m,
        "intra_over_inter": intra_m / max(inter_m, 1e-8),
    }


@torch.no_grad()
def reconstruction_report(
    cvae: EmbeddingCVAE,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    device = next(cvae.parameters()).device
    x = _normalize(embeddings).to(device)
    y = labels.view(-1).long().to(device)
    recon, mu, logvar, z = cvae(x, y)
    cos = (x * recon).sum(dim=-1)
    return {
        "recon_cosine_mean": cos.mean().item(),
        "recon_cosine_std": cos.std().item(),
        "recon_1_minus_cos": (1.0 - cos).mean().item(),
        "kl": EmbeddingCVAE.kl_loss(mu, logvar).item(),
        "latent_mu_norm": mu.norm(dim=-1).mean().item(),
        "latent_std_mean": torch.exp(0.5 * logvar).mean().item(),
    }


@torch.no_grad()
def latent_class_separation(
    cvae: EmbeddingCVAE, embeddings: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    device = next(cvae.parameters()).device
    x = _normalize(embeddings).to(device)
    y = labels.view(-1).long().to(device)
    mu, _ = cvae.encode(x, y)
    protos = class_prototypes(mu, y)

    classes = sorted(protos.keys())
    if len(classes) < 2:
        return {"latent_intra_l2": 0.0, "latent_inter_l2": 0.0, "latent_fisher_ratio": 0.0}

    intra, inter = [], []
    for i, ci in enumerate(classes):
        mask_i = y == ci
        mi = mu[mask_i]
        pi = protos[ci]
        intra.extend((mi - pi.unsqueeze(0)).norm(dim=-1).tolist())
        for cj in classes[i + 1 :]:
            inter.append((pi - protos[cj]).norm().item())

    intra_m = sum(intra) / max(len(intra), 1)
    inter_m = sum(inter) / max(len(inter), 1)
    return {
        "latent_intra_l2": intra_m,
        "latent_inter_l2": inter_m,
        "latent_fisher_ratio": inter_m / max(intra_m, 1e-8),
    }


@torch.no_grad()
def compare_sources(
    real: torch.Tensor,
    real_labels: torch.Tensor,
    candidate: torch.Tensor,
    candidate_labels: torch.Tensor,
    name: str,
) -> Dict[str, float]:
    """
    Compare candidate replay (generated / buffer) to real class structure.
    Uses prototypes built from REAL train embeddings only.
    """
    protos = class_prototypes(real, real_labels)
    real_proto_acc = prototype_accuracy(real, real_labels, protos)
    cand_proto_acc = prototype_accuracy(candidate, candidate_labels, protos)
    real_cos, _ = mean_cosine_to_prototypes(real, real_labels, protos)
    cand_cos, _ = mean_cosine_to_prototypes(candidate, candidate_labels, protos)

    # Real-to-real NN retrieval @1 (upper bound)
    real_nn = _retrieval_at_k(real, real_labels, real, real_labels, k=1)
    cand_nn = _retrieval_at_k(candidate, candidate_labels, real, real_labels, k=1)

    out = {
        f"{name}_proto_acc": cand_proto_acc,
        f"{name}_mean_cos_to_real_proto": cand_cos,
        f"{name}_retrieval_at_1": cand_nn,
        "real_proto_acc": real_proto_acc,
        "real_mean_cos_to_real_proto": real_cos,
        "real_retrieval_at_1": real_nn,
        f"{name}_proto_acc_gap_vs_real": real_proto_acc - cand_proto_acc,
        f"{name}_cos_gap_vs_real": real_cos - cand_cos,
    }
    out.update({f"{name}_{k}": v for k, v in intra_inter_cosine(candidate, candidate_labels).items()})
    return out


@torch.no_grad()
def _retrieval_at_k(
    query: torch.Tensor,
    query_labels: torch.Tensor,
    gallery: torch.Tensor,
    gallery_labels: torch.Tensor,
    k: int = 1,
) -> float:
    if query.numel() == 0 or gallery.numel() == 0:
        return 0.0
    q = _normalize(query)
    g = _normalize(gallery)
    ql = query_labels.view(-1).long()
    gl = gallery_labels.view(-1).long()
    sims = q @ g.T
    topk = sims.topk(min(k, g.size(0)), dim=1).indices
    hits = 0
    for i in range(q.size(0)):
        if (gl[topk[i]] == ql[i]).any():
            hits += 1
    return hits / q.size(0)


def format_metrics_table(rows: List[Dict[str, float]], title: str) -> str:
    if not rows:
        return f"\n{title}\n(empty)\n"
    keys = sorted({k for r in rows for k in r.keys()})
    lines = [f"\n{'=' * len(title)}", title, "=" * len(title)]
    header = " | ".join(f"{k:>22}" for k in keys)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(" | ".join(f"{r.get(k, float('nan')):>22.4f}" for k in keys))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generation quality: compare CVAE samples to a real reference buffer
# ---------------------------------------------------------------------------


@torch.no_grad()
def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Tuple[float, ...] = (0.5, 1.0, 5.0),
) -> float:
    """
    Unbiased MMD² with multi-scale RBF kernel.

    Lower is better (0 = identical distributions).
    Operates on CPU float tensors; subsamples to *max_n* rows for speed.
    """
    max_n = 256
    if x.size(0) > max_n:
        x = x[torch.randperm(x.size(0))[:max_n]]
    if y.size(0) > max_n:
        y = y[torch.randperm(y.size(0))[:max_n]]

    x = x.float().cpu()
    y = y.float().cpu()
    n, m = x.size(0), y.size(0)
    if n < 2 or m < 2:
        return float("nan")

    def _rbf_sum(a: torch.Tensor, b: torch.Tensor) -> float:
        diff = a.unsqueeze(1) - b.unsqueeze(0)   # (n, m, d)
        sq = diff.pow(2).sum(-1)                  # (n, m)
        return sum(torch.exp(-sq / (2 * s ** 2)).sum().item() for s in sigmas)

    xx = _rbf_sum(x, x)
    yy = _rbf_sum(y, y)
    xy = _rbf_sum(x, y)

    scale = len(sigmas)
    mmd2 = (
        (xx - x.size(0) * scale) / (n * (n - 1))
        + (yy - y.size(0) * scale) / (m * (m - 1))
        - 2 * xy / (n * m)
    )
    return float(mmd2)


@torch.no_grad()
def per_class_generation_report(
    real_by_class: Dict[int, torch.Tensor],
    gen_by_class: Dict[int, torch.Tensor],
) -> Dict[str, Any]:
    """
    Compare CVAE-generated embeddings against a real reference buffer.

    For each class *c* that has reference data, computes:

    ``centroid_l2``
        L2 distance between real and generated centroids.  0 = perfect.
    ``std_ratio``
        Mean(std_generated) / Mean(std_real) per feature, then averaged.
        1.0 = same spread; < 1 = mode collapse; > 1 = over-dispersion.
    ``cosine_sim``
        Cosine similarity between real and generated centroids.  1.0 = perfect.
    ``fd``
        Diagonal Fréchet distance: ||μ_r − μ_g||² + ||σ_r − σ_g||².
        Lower is better.
    ``mmd``
        Unbiased MMD² with multi-scale RBF kernel.  Lower is better.

    Returns a dict with keys ``"per_class"`` (per-class dicts) and ``"agg"``
    (aggregate summary).  Classes missing from *gen_by_class* or with < 2
    samples in either set are marked ``None``.
    """
    per_class: Dict[int, Optional[Dict[str, float]]] = {}

    for c in sorted(real_by_class):
        real = real_by_class[c]
        gen = gen_by_class.get(c)

        if (
            gen is None
            or gen.numel() == 0
            or real.numel() == 0
            or real.dim() < 2
            or gen.dim() < 2
            or real.size(0) < 2
            or gen.size(0) < 2
        ):
            per_class[c] = None
            continue

        real = real.float()
        gen = gen.float()

        mu_r = real.mean(0)
        mu_g = gen.mean(0)
        std_r = real.std(0).clamp(min=1e-8)
        std_g = gen.std(0).clamp(min=1e-8)

        centroid_l2 = float((mu_r - mu_g).norm().item())
        std_ratio = float((std_g / std_r).mean().item())
        cos_sim = float(F.cosine_similarity(mu_r.unsqueeze(0), mu_g.unsqueeze(0)).item())
        fd = float((mu_r - mu_g).pow(2).sum().item() + (std_r - std_g).pow(2).sum().item())
        mmd = mmd_rbf(real, gen)

        per_class[c] = {
            "centroid_l2": centroid_l2,
            "std_ratio": std_ratio,
            "cosine_sim": cos_sim,
            "fd": fd,
            "mmd": mmd,
        }

    valid = [v for v in per_class.values() if v is not None]
    if not valid:
        return {"per_class": per_class, "agg": {}}

    def _mean(key: str) -> float:
        vals = [v[key] for v in valid if not (isinstance(v[key], float) and v[key] != v[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    worst_fd_cls = max((c for c, v in per_class.items() if v is not None), key=lambda c: per_class[c]["fd"])  # type: ignore[index]
    worst_mmd_cls = max((c for c, v in per_class.items() if v is not None), key=lambda c: per_class[c]["mmd"])  # type: ignore[index]

    agg: Dict[str, Any] = {
        "mean_centroid_l2": _mean("centroid_l2"),
        "mean_std_ratio": _mean("std_ratio"),
        "mean_cosine_sim": _mean("cosine_sim"),
        "mean_fd": _mean("fd"),
        "mean_mmd": _mean("mmd"),
        "worst_class_fd": per_class[worst_fd_cls]["fd"],    # type: ignore[index]
        "worst_class_fd_id": worst_fd_cls,
        "worst_class_mmd": per_class[worst_mmd_cls]["mmd"],  # type: ignore[index]
        "worst_class_mmd_id": worst_mmd_cls,
        "n_classes_measured": len(valid),
    }
    return {"per_class": per_class, "agg": agg}


def format_generation_quality_table(
    report: Dict[str, Any],
    task_id: int,
    current_task_classes: Optional[List[int]] = None,
    top_k_worst: int = 5,
) -> str:
    """
    Pretty-print a generation quality report.

    Shows aggregate numbers, then the *top_k_worst* classes (highest FD) so
    that forgetting hot-spots are immediately visible.
    """
    agg = report.get("agg", {})
    per_class = report.get("per_class", {})

    if not agg:
        return f"\n[CVAE quality | task {task_id + 1}] (no data)\n"

    current = set(current_task_classes or [])
    old = [c for c in per_class if c not in current and per_class[c] is not None]
    new = [c for c in per_class if c in current and per_class[c] is not None]

    lines = [
        "",
        f"{'─' * 72}",
        f"  CVAE generation quality  [after task {task_id + 1}]",
        f"{'─' * 72}",
        f"  classes measured : {agg['n_classes_measured']:>4}  "
        f"(current={len(new)}, old={len(old)})",
        f"  mean centroid L2 : {agg['mean_centroid_l2']:>8.4f}   "
        "(↓ better)",
        f"  mean std ratio   : {agg['mean_std_ratio']:>8.4f}   "
        "(≈1.0 ideal; <1 = collapse; >1 = over-dispersion)",
        f"  mean cosine sim  : {agg['mean_cosine_sim']:>8.4f}   "
        "(↑ better, 1.0 = perfect)",
        f"  mean FD (diag)   : {agg['mean_fd']:>8.4f}   "
        "(↓ better)",
        f"  mean MMD²        : {agg['mean_mmd']:>8.6f}   "
        "(↓ better)",
        f"  worst FD  → class {agg['worst_class_fd_id']:>3}  "
        f"FD={agg['worst_class_fd']:.4f}",
        f"  worst MMD → class {agg['worst_class_mmd_id']:>3}  "
        f"MMD={agg['worst_class_mmd']:.6f}",
    ]

    # Show worst-k old classes by FD (most likely to be forgotten)
    if old:
        worst_old = sorted(old, key=lambda c: per_class[c]["fd"], reverse=True)[:top_k_worst]  # type: ignore[index]
        lines.append(f"  {'─' * 66}")
        lines.append(f"  Worst {len(worst_old)} OLD classes (highest FD — potential forgetting):")
        lines.append(
            f"  {'cls':>5} {'centroid_L2':>12} {'std_ratio':>10} "
            f"{'cos_sim':>8} {'FD':>10} {'MMD':>10}"
        )
        for c in worst_old:
            v = per_class[c]
            lines.append(
                f"  {c:>5} {v['centroid_l2']:>12.4f} {v['std_ratio']:>10.4f} "  # type: ignore[index]
                f"{v['cosine_sim']:>8.4f} {v['fd']:>10.4f} {v['mmd']:>10.6f}"  # type: ignore[index]
            )

    lines.append(f"{'─' * 72}")
    return "\n".join(lines)
