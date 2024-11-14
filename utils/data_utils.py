from typing import Any, Dict, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from incremental_dataloaders.datasets import (
    BaseDataset,
    CIFAR100Dataset,
    CIFAR10Dataset,
    MNISTDataset,
    TinyImageNetDataset,
)

DATASET_CLASSES = {
    "MNIST": MNISTDataset,
    "CIFAR10": CIFAR10Dataset,
    "CIFAR100": CIFAR100Dataset,
    "TinyImageNet": TinyImageNetDataset,
}


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


def set_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    transform = get_transforms(config["dataset"], config["flatten"])
    train_dataset, test_dataset, num_classes = get_datasets(
        config["dataset"], transform
    )

    config["num_classes"] = num_classes

    # Use the specified number of instances, or all if there are fewer
    num_train = min(config["num_instances"], len(train_dataset))
    num_test = min(config["num_instances"] // 5, len(test_dataset))

    train_indices = torch.randperm(len(train_dataset))[:num_train]
    test_indices = torch.randperm(len(test_dataset))[:num_test]

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=config["batch_size"], shuffle=False
    )

    print(
        f"Using {num_train} instances for training and {num_test} instances for testing"
    )

    return train_loader, test_loader
