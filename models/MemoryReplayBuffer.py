import random
from typing import Optional, Tuple
import torch
from collections import defaultdict, deque


class MemoryReplayBuffer:
    def __init__(self, classes_per_task: int, memory_size_per_class: Optional[int]):
        self.buffer = defaultdict(lambda: deque(maxlen=memory_size_per_class))
        self.class_count = 0
        self.classes_per_task = classes_per_task

    def add_task_samples(self, x: torch.Tensor, y: torch.Tensor, task: int) -> None:
        x_cpu = x.cpu()

        current_classes = set(
            range(self.classes_per_task * task, (task + 1) * self.classes_per_task)
        )

        for i, label in enumerate(y):
            label_item = label.item()

            if label_item not in current_classes:
                continue

            self.buffer[label_item].append((x_cpu[i], label))
            if label >= self.class_count:
                self.class_count = label_item + 1

    def get_memory_samples(self, batch_size: int = None) -> Tuple[torch.Tensor]:
        x_memory, y_memory = [], []

        if self.class_count == 0:
            return torch.empty(0), torch.empty(0)

        if batch_size is None:
            for samples in self.buffer.values():
                x_memory.extend([x for x, _ in samples])
                y_memory.extend([y for _, y in samples])
        else:
            samples_per_class = max(1, (batch_size // self.class_count))
            for samples in self.buffer.values():
                if len(samples) > 0:
                    sampled = random.sample(
                        samples, min(samples_per_class, len(samples))
                    )
                    x_memory.extend([x for x, _ in sampled])
                    y_memory.extend([y for _, y in sampled])

        if x_memory:  # Check if the list is not empty
            x_memory = torch.stack(x_memory)
            y_memory = torch.stack(y_memory)
            if self.class_count is not None:
                y_memory = self._expand_one_hot(y_memory, self.class_count)
        else:
            x_memory = torch.empty(0)
            y_memory = torch.empty(0)

        return x_memory, y_memory

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

    def _expand_one_hot(self, one_hot: torch.Tensor, num_classes: int):
        current_classes = one_hot.size(1)
        if num_classes > current_classes:
            padding = torch.full((one_hot.size(0), num_classes - current_classes), 0.05)
            return torch.cat([one_hot, padding], dim=1)
        return one_hot

    def get_class_distribution(self):
        return {label: len(samples) for label, samples in self.buffer.items()}

    def clear(self):
        self.buffer.clear()
        self.class_count = 0

    def get_num_classes(self):
        return self.class_count
