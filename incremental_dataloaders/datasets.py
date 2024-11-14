import os
from typing import Tuple, Union
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class BaseDataset(Dataset):
    def __init__(self, root: str, train: bool, img_size: int):
        self.root = root
        self.train = train
        self.dataset = None
        self.transform = self.get_transform(img_size=img_size)

    def load_data(self):
        """Abstract method to be implemented by each subclass to load its specific data."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_transform(self, img_size: Union[Tuple, int]):
        """Abstract method to be implemented by each subclass for dataset-specific transforms."""
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class MNISTDataset(BaseDataset):
    def __init__(self, root="./data", train=True, img_size: int = 224):
        super().__init__(root, train, img_size=img_size)
        self.dataset = datasets.MNIST(
            root=self.root, train=self.train, download=True, transform=self.transform
        )

    def get_transform(self, img_size: Union[Tuple, int]):
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )


class CIFAR10Dataset(BaseDataset):
    def __init__(self, root="./data", train=True, img_size: int = 224):
        super().__init__(root, train, img_size=img_size)
        self.dataset = datasets.CIFAR10(
            root=self.root, train=self.train, download=True, transform=self.transform
        )

    def get_transform(self, img_size: Union[Tuple, int]):
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )


class CIFAR100Dataset(BaseDataset):
    def __init__(self, root="./data", train=True, img_size: int = 224):
        super().__init__(root, train, img_size=img_size)
        self.dataset = datasets.CIFAR100(
            root=self.root, train=self.train, download=True, transform=self.transform
        )

    def get_transform(self, img_size: Union[Tuple, int]):
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )


class TinyImageNetDataset(BaseDataset):
    def __init__(self, root="./data", train=True, img_size: int = 224):
        root = os.path.join(root, "tiny-imagenet", "train" if train else "val")
        super().__init__(root, train, img_size=img_size)
        self.dataset = datasets.ImageFolder(root=self.root, transform=self.transform)

    def get_transform(self, img_size: Union[Tuple, int]):
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
