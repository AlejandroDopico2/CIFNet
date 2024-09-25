from typing import Any, Dict, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets


def get_transforms(dataset: str, flatten: bool) -> transforms.Compose:

    if dataset == "MNIST":
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    elif dataset.startswith("CIFAR"):
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    else:
        raise ValueError("Unsupported dataset")

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    return transforms.Compose(transform_list)


def get_datasets(
    dataset: str, transform: transforms.Compose, binary: bool
) -> Tuple[Dataset, Dataset]:

    if dataset == "MNIST":
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        num_classes = 10
    elif dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        num_classes = 10
    elif dataset == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
        num_classes = 100
    else:
        raise ValueError("Unsupported dataset")

    if binary:
        train_indices = (train_dataset.targets == 0) | (train_dataset.targets == 1)
        train_dataset = Subset(train_dataset, torch.where(train_indices)[0])

        test_indices = (test_dataset.targets == 0) | (test_dataset.targets == 1)
        test_dataset = Subset(test_dataset, torch.where(test_indices)[0])
        num_classes = 2

    return train_dataset, test_dataset, num_classes


def set_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    transform = get_transforms(config["dataset"], config["flatten"])
    train_dataset, test_dataset, num_classes = get_datasets(
        config["dataset"], transform, config["binary"]
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
