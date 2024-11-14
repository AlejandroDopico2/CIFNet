from typing import Dict, List, Optional, Tuple
import torch
from collections import defaultdict

from models.samplers import SamplingStrategy


class MemoryExpansionBuffer:
    def __init__(
        self,
        classes_per_task: int,
        memory_size_per_class: Optional[int],
        sampling_strategy: SamplingStrategy,
    ):
        self.buffer: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.memory_size_per_class = memory_size_per_class
        self.class_count = 0
        self.classes_per_task = classes_per_task
        self.sampling_strategy = sampling_strategy

    def add_task_samples(self, x: torch.Tensor, y: torch.Tensor, task: int) -> None:
        current_classes = set(
            range(self.classes_per_task * task, (task + 1) * self.classes_per_task)
        )

        for sample, label in zip(x, y):
            label_item = torch.argmax(label).item()

            if label_item not in current_classes:
                continue

            self.buffer[label_item].append(sample.cpu())
            if label_item >= self.class_count:
                self.class_count = label_item + 1

    def sample(self, **kwargs):
        for label, samples in self.buffer.items():
            if not isinstance(self.buffer[label], torch.Tensor):
                self.buffer[label] = torch.stack(samples)

        self.buffer = self.sampling_strategy.sample(
            buffer=self.buffer, n_samples=self.memory_size_per_class, **kwargs
        )

    def get_memory_samples(
        self, classes: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor]:
        x_memory, y_memory = [], []

        if classes:
            for label in classes:
                if label not in self.buffer:
                    raise ValueError("Label specified not in the buffer.")

                x_memory.extend(self.buffer[label])
                y_memory.extend([label] * len(self.buffer[label]))
        else:
            for label, samples in self.buffer.items():
                x_memory.extend(samples)
                y_memory.extend([label] * len(samples))

        return torch.stack(x_memory), torch.LongTensor(y_memory)

    def get_past_tasks_samples(
        self, task_id: int, batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor]:
        x_memory, y_memory = [], []

        if task_id >= self.class_count:
            return torch.empty(0), torch.empty(0)

        class_range = range(task_id * self.classes_per_task)
        if batch_size is None:
            for class_id in class_range:
                x_memory.extend([x for x, _ in self.buffer[class_id]])
                y_memory.extend([y for _, y in self.buffer[class_id]])

        # else: TODO Add dynamic buffer size

        if x_memory:  # Check if the list is not empty
            x_memory = torch.stack(x_memory)
            y_memory = torch.tensor(y_memory, dtype=torch.long)
        else:
            x_memory = torch.empty(0)
            y_memory = torch.empty(0)

        if not torch.is_tensor(x_memory) or not torch.is_tensor(y_memory):
            raise ValueError("Input must be of type torch.Tensor")

        return x_memory, y_memory

    def _one_hot_encode(self, labels: torch.Tensor, num_classes: int = None):
        if num_classes is None:
            num_classes = self.class_count
        return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

    def _expand_one_hot(self, y: torch.Tensor, num_classes: int):
        current_classes = y.unique().size(0)
        if num_classes > current_classes:
            padding = torch.full((y.size(0), num_classes - current_classes), 0.05)
            return torch.cat([y, padding], dim=1)
        return y

    def get_class_distribution(self):
        return {label: len(samples) for label, samples in self.buffer.items()}

    def clear(self):
        self.buffer.clear()
        self.class_count = 0

    def get_num_classes(self):
        return self.class_count
