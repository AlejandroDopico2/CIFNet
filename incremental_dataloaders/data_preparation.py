import torch
import numpy as np
from typing import List, Optional, Tuple
from torch.utils.data import Dataset

from incremental_dataloaders.custom_datasets import TensorSubset
from incremental_dataloaders.datasets import (
    BaseDataset,
    CIFAR100Dataset,
    CIFAR10Dataset,
    ImageNet100Dataset,
    MNISTDataset,
    TinyImageNetDataset,
)

DATASET_CLASSES = {
    "MNIST": MNISTDataset,
    "CIFAR10": CIFAR10Dataset,
    "CIFAR100": CIFAR100Dataset,
    "TinyImageNet": TinyImageNetDataset,
    "ImageNet100": ImageNet100Dataset,
}


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


def get_dataset_instance(
    dataset_name: str, root: str = "./data", img_size: int = 224
) -> Tuple[BaseDataset, BaseDataset]:
    """
    Instantiate and return train and test dataset instances based on dataset name.

    Args:
        dataset_name (str): The name of the dataset.
        root (str): The root directory for storing dataset files.
        img_size (int): Image size to be used in transforms.

    Returns:
        Tuple[BaseDataset, BaseDataset]: Train and test dataset instances.
    """
    # Retrieve the dataset class from the dictionary
    dataset_class = DATASET_CLASSES.get(dataset_name)

    if not dataset_class:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    # Instantiate the dataset for train and test
    train_dataset = dataset_class(root=root, train=True, img_size=img_size)
    test_dataset = dataset_class(root=root, train=False, img_size=img_size)

    return train_dataset, test_dataset
