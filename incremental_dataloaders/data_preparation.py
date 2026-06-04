import torch
import numpy as np
from typing import List, Optional, Tuple
from torch.utils.data import Dataset
from loguru import logger
from incremental_dataloaders.custom_datasets import TensorSubset
from incremental_dataloaders.datasets import (
    BaseDataset,
    CIFAR100Dataset,
    CIFAR10Dataset,
    ImageFolderDataset,
    MNISTDataset,
    TinyImageNetDataset,
)

DATASET_CLASSES = {
    "MNIST": MNISTDataset,
    "CIFAR10": CIFAR10Dataset,
    "CIFAR100": CIFAR100Dataset,
    "TinyImageNet": TinyImageNetDataset,
    "ImageFolder": ImageFolderDataset,
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
    dataset_name: str,
    root: str = "./data",
    img_size: int = 224,
    backbone: Optional[str] = None,
    use_vit: Optional[bool] = None,
    use_clip: Optional[bool] = None,
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

    backbone_name = (backbone or "").lower()
    if use_clip is None:
        use_clip = "clip" in backbone_name
    if use_vit is None:
        use_vit = "vit" in backbone_name and not use_clip

    train_dataset = dataset_class(
        root=root, train=True, img_size=img_size, use_vit=use_vit, use_clip=use_clip
    )
    test_dataset = dataset_class(
        root=root, train=False, img_size=img_size, use_vit=use_vit, use_clip=use_clip
    )

    return train_dataset, test_dataset
