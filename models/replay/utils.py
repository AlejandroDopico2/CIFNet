from collections import defaultdict
from typing import Tuple

import torch


def upsample_per_class(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    samples_per_class: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repeat (or subsample) so each class has exactly samples_per_class rows."""
    if embeddings.numel() == 0 or samples_per_class <= 0:
        return embeddings, labels

    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    labels = labels.view(-1).long()

    parts_x, parts_y = [], []
    for c in labels.unique().tolist():
        c = int(c)
        mask = labels == c
        cls_x = embeddings[mask]
        n = cls_x.size(0)
        if n == 0:
            continue
        if n >= samples_per_class:
            idx = torch.randperm(n)[:samples_per_class]
            cls_x = cls_x[idx]
        else:
            reps = samples_per_class // n
            rem = samples_per_class % n
            cls_x = torch.cat(
                [cls_x] * reps + ([cls_x[:rem]] if rem else []),
                dim=0,
            )
        parts_x.append(cls_x)
        parts_y.append(torch.full((cls_x.size(0),), c, dtype=torch.long))

    if not parts_x:
        return embeddings, labels

    return torch.cat(parts_x, dim=0), torch.cat(parts_y, dim=0)
