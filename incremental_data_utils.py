from typing import Any, Dict, Tuple
import numpy as np
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
    dataset: str, transform: transforms.Compose, binary: bool = False
) -> Tuple[Dataset, Dataset]:

    if dataset == "MNIST":
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError("Unsupported dataset")

    if binary:
        train_indices = (train_dataset.targets == 0) | (train_dataset.targets == 1)
        train_dataset = Subset(train_dataset, torch.where(train_indices)[0])

        test_indices = (test_dataset.targets == 0) | (test_dataset.targets == 1)
        test_dataset = Subset(test_dataset, torch.where(test_indices)[0])

    return train_dataset, test_dataset


def prepare_data(dataset: Dataset, class_range: int, samples_per_task: int) -> Subset:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise AttributeError("Dataset must have an attribute 'targets' or 'labels'.")
    
    samples_per_class = samples_per_task // len(class_range)
    
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    targets = torch.IntTensor(targets) if not isinstance(targets, torch.Tensor) else targets

    class_indices = [torch.where(targets == i)[0] for i in class_range]

    selected_indices = torch.cat(
        [
            torch.tensor(
                np.random.choice(indices.numpy(), samples_per_class, replace=True)
            )
            for indices in class_indices
        ]
    )

    return Subset(dataset, selected_indices)


if __name__ == "__main__":

    train_dataset, test_dataset, num_classes = get_datasets(
        "MNIST", get_transforms("MNIST", False)
    )

    classes_per_task = 2
    for i in range(1, 6):
        task = i
        class_range = list(
            range((task - 1) * classes_per_task, task * classes_per_task)
        )

        dataset = prepare_data(train_dataset, class_range, samples_per_class=100)

        print("Task", i, torch.unique(torch.IntTensor([y for _, y in dataset])))
