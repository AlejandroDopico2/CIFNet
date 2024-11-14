import torch
import numpy as np
from typing import List, Optional
from torch.utils.data import Dataset

from incremental_dataloaders.custom_datasets import TensorSubset


def prepare_data(
    dataset: Dataset, class_range: List[int], samples_per_task: Optional[int] = None
) -> TensorSubset:
    targets = getattr(
        dataset.dataset, "targets", getattr(dataset.dataset, "labels", None)
    )
    if targets is None:
        raise AttributeError("Dataset must have an attribute 'targets' or 'labels'.")

    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    elif isinstance(targets, list):
        targets = torch.from_numpy(np.array(targets))
    else:
        targets = targets.clone()

    targets = targets.long()

    if samples_per_task is None:
        class_indices = torch.cat([torch.where(targets == i)[0] for i in class_range])
    else:
        samples_per_class = samples_per_task // len(class_range)
        class_indices = []
        for i in class_range:
            class_indices_i = torch.where(targets == i)[0]
            selected_indices = class_indices_i[
                torch.randperm(len(class_indices_i))[:samples_per_class]
            ]
            class_indices.append(selected_indices)
        class_indices = torch.cat(class_indices)

    return TensorSubset(dataset, class_indices)
