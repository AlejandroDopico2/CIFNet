import os
import struct
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.targets = y

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.X[idx], self.targets[idx]


def load_mnist(path, kind="train", flatten: bool = False):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte")

    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8)
        if flatten:
            images = images.reshape(len(labels), 784)
        else:
            images = images.reshape(len(labels), 28, 28)
            images = np.moveaxis(images, -1, 1)
            images = np.expand_dims(images, 1)
        images = images.astype(np.float32) / 255.0

    return images, labels


def get_transforms(dataset: str, flatten: bool) -> transforms.Compose:
    transform_list = []

    if flatten:
        transform_list = [transforms.Lambda(lambda x: x.view(-1))]
    else:
        transform_list = [transforms.Resize((224, 224), interpolation=Image.LANCZOS)]

    if dataset == "MNIST":
        transform_list.extend(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    elif dataset.startswith("CIFAR"):
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
    else:
        raise ValueError("Unsupported dataset")

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
    elif dataset == "PLACES":
        train_dataset = datasets.Places365(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.Places365(
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


class TensorSubset(Subset):
    def __getitem__(self, index):
        x, y = self.dataset[self.indices[index]]
        return x, torch.tensor(y) if not isinstance(y, torch.Tensor) else y


def prepare_data(
    dataset: Dataset, class_range: List[int], samples_per_task: Optional[int] = None
) -> Subset:
    # Determine whether the dataset uses 'targets' or 'labels'
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise AttributeError("Dataset must have an attribute 'targets' or 'labels'.")

    # Convert targets to a PyTorch tensor if it's not already
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    elif not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # Ensure targets are integers
    targets = targets.long()

    if samples_per_task is None:
        # If no sample limit, select all instances of the specified classes
        class_indices = torch.cat([torch.where(targets == i)[0] for i in class_range])
    else:
        samples_per_class = samples_per_task // len(class_range)
        class_indices = []
        for i in class_range:
            class_mask = targets == i
            class_indices_i = torch.where(class_mask)[0]

            if len(class_indices_i) > samples_per_class:
                selected_indices = torch.randperm(len(class_indices_i))[
                    :samples_per_class
                ]
                class_indices.append(class_indices_i[selected_indices])
            else:
                class_indices.append(class_indices_i)
        class_indices = torch.cat(class_indices)

    return TensorSubset(dataset, class_indices)


def get_class_instances(
    dataset: Dataset, class_range: int, samples_per_task: Optional[int] = None
) -> Subset:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise AttributeError("Dataset must have an attribute 'targets' or 'labels'.")

    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    targets = (
        torch.IntTensor(targets) if not isinstance(targets, torch.Tensor) else targets
    )

    class_indices = list(class_range)

    all_targets = torch.tensor([dataset.targets[i] for i in range(len(dataset))])

    masks = [all_targets == cls for cls in class_indices]

    combined_mask = torch.logical_or(*masks)

    selected_indices = combined_mask.nonzero().squeeze()

    if samples_per_task is not None:
        final_indices = []
        for cls in class_indices:
            cls_mask = all_targets[selected_indices] == cls
            cls_indices = selected_indices[cls_mask]

            n_samples = min(samples_per_task, len(cls_indices))
            selected = torch.randperm(len(cls_indices))[:n_samples]
            final_indices.append(cls_indices[selected])

        selected_indices = torch.cat(final_indices)

    selected_X = [dataset.X[i] for i in selected_indices]
    selected_targets = [dataset.targets[i] for i in selected_indices]

    return CustomDataset(selected_X, selected_targets)


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
