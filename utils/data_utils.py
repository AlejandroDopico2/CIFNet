from typing import Tuple

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
