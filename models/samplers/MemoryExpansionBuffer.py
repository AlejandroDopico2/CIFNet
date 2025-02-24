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
        self.buffer: Dict[int, torch.Tensor] = defaultdict(lambda: torch.empty(0))
        self.memory_size = memory_size_per_class
        self.classes_per_task = classes_per_task
        self.sampling_strategy = sampling_strategy
        self.current_max_class = 0

    def add_task_samples(self, embeddings: torch.Tensor, labels: torch.Tensor, task: int) -> None:
        """Add samples to buffer using vectorized operations"""
        class_indices = torch.argmax(labels, dim=1)

        task_start = task * self.classes_per_task
        task_end = (task + 1) * self.classes_per_task
        task_classes = range(task_start, task_end)

        for task_class in task_classes:
            mask = (class_indices == task_class)
            task_class_embeddings = embeddings[mask].cpu()
            
            if task_class_embeddings.size(0) == 0:
                continue

            self.buffer[task_class] = torch.cat([self.buffer[task_class], task_class_embeddings])
            self.current_max_class = max(self.current_max_class, task_class + 1)

        self._maintain_buffer()

    def _maintain_buffer(self):
        """Apply sampling strategy to maintain buffer size per class"""
        for cls in list(self.buffer.keys()):
            if self.buffer[cls].size(0) > self.memory_size:
                self.buffer[cls] = self.sampling_strategy.sample(
                    self.buffer[cls], 
                    self.memory_size
                )

    def get_memory_samples(self, classes: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficiently retrieve samples with tensor concatenation"""
        valid_classes = [c for c in classes if c in self.buffer]
        
        if not valid_classes:
            return torch.empty(0), torch.empty(0)

        embeddings = torch.cat([self.buffer[c] for c in valid_classes])

        labels = torch.cat([torch.full(size=len(self.buffer[c]), fill_value=c, dtype=torch.long) for c in valid_classes])

        return embeddings, labels

    def get_class_distribution(self) -> Dict[int, int]:
        return {c: len(emb) for c, emb in self.buffer.items()}

    def __len__(self) -> int:
        return sum(e.size(0) for e in self.buffer.values())
