from typing import Dict, List, Optional, Tuple
import torch
from collections import defaultdict
from models.samplers import SamplingStrategy


class MemoryExpansionBuffer:
    def __init__(
        self,
        total_memory_size: int,
        sampling_strategy: SamplingStrategy,
    ):
        self.buffer: Dict[int, torch.Tensor] = defaultdict(lambda: torch.empty(0))
        self.total_memory_size = total_memory_size
        self.sampling_strategy = sampling_strategy
        self.current_max_class = 0

    def add_task_samples(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Add samples to buffer using vectorized operations"""
        unique_classes = labels.unique().tolist()

        for task_class in unique_classes:
            mask = labels == task_class
            task_class_embeddings = embeddings[mask].cpu().squeeze(0)

            if task_class_embeddings.size(0) == 0:
                continue

            if task_class_embeddings.ndim == 1:
                task_class_embeddings = task_class_embeddings.unsqueeze(0)

            self.buffer[task_class] = torch.cat(
                [self.buffer[task_class], task_class_embeddings]
            )
            self.current_max_class = max(self.current_max_class, task_class + 1)

    def _maintain_buffer(self):
        """
        Maintain a strictly balanced buffer across classes.
        """
        num_classes = len(self.buffer)
        if num_classes == 0:
            return

        per_class_quota = self.total_memory_size // num_classes

        for cls in list(self.buffer.keys()):
            if self.buffer[cls].size(0) > per_class_quota:
                self.buffer[cls] = self.sampling_strategy.sample(
                    self.buffer[cls], per_class_quota
                )

    def get_memory_samples(
        self, classes: List[int], buffer_state: Optional[Dict[int, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficiently retrieve samples with tensor concatenation"""
        source_buffer = buffer_state if buffer_state is not None else self.buffer

        valid_classes = [c for c in classes if c in source_buffer]

        if not valid_classes:
            return torch.empty(0), torch.empty(0)

        embeddings = torch.cat([source_buffer[c] for c in valid_classes])

        labels = torch.cat(
            [
                torch.full(
                    size=(len(source_buffer[c]),), fill_value=c, dtype=torch.long
                )
                for c in valid_classes
            ]
        )

        return embeddings, labels

    def get_class_distribution(
        self, buffer: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[int, int]:
        return {
            c: len(emb)
            for c, emb in (buffer if buffer is not None else self.buffer).items()
        }

    def get_buffer_state(self) -> Dict[int, torch.Tensor]:
        """Returns a deep copy of the current buffer state."""
        return {cls: tensor.clone() for cls, tensor in self.buffer.items()}

    def __len__(self) -> int:
        return sum(e.size(0) for e in self.buffer.values())

    def get_classes_stored(self) -> List[int]:
        return list(self.buffer.keys())

    def get_max_samples_per_class(self) -> int:
        """Returns the maximum number of samples stored for any class in the buffer."""
        if not self.buffer:
            return 0
        return max(tensor.size(0) for tensor in self.buffer.values())