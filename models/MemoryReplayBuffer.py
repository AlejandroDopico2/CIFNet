import random
import torch
from collections import defaultdict, deque
import numpy as np

class MemoryReplayBuffer:
    def __init__(self, memory_size_per_class: int, store_targets: bool = True):
        self.memory_size_per_class = memory_size_per_class
        self.buffer = defaultdict(lambda: deque(maxlen = memory_size_per_class))
        self.store_targets = store_targets
        self.class_count = 0
        
    def add_samples(self, x: torch.Tensor, y: torch.Tensor):
        if self.store_targets:
            labels = y.argmax(dim=1).cpu()
        else:
            labels = y.argmax(dim=1).cpu()
            y = y.cpu()

        x_cpu = x.cpu()
        
        for i, label in enumerate(labels):
            label_item = label.item()
            self.buffer[label_item].append((x_cpu[i], y[i] if not self.store_targets else label))
            if label >= self.class_count:
                self.class_count = label_item + 1
        
    def get_memory_samples(self, batch_size: int = None, num_classes: int = None) -> torch.Tensor:
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
                    sampled = random.sample(samples, min(samples_per_class, len(samples)))
                    x_memory.extend([x for x, _ in sampled])
                    y_memory.extend([y for _, y in sampled])


        if x_memory:  # Check if the list is not empty
            x_memory = torch.stack(x_memory)
            if self.store_targets:
                y_memory = torch.tensor(y_memory, dtype=torch.long)
                y_memory = self._one_hot_encode(y_memory, num_classes)
            else:
                y_memory = torch.stack(y_memory)
                if num_classes is not None:
                    y_memory = self._expand_one_hot(y_memory, num_classes)
        else:
            x_memory = torch.empty(0)
            y_memory = torch.empty(0)
        
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