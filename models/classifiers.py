from loguru import logger
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import SGDClassifier
from typing import List, Optional

from models.ROLANN import ROLANN


class RolannClassifier(ROLANN):
    """
    Thin wrapper around the original ROLANN implementation so it matches
    the incremental classifier interface expected by CIFNet.
    """

    def aggregate_update(self, X: torch.Tensor, d: torch.Tensor, classes: Optional[torch.Tensor]) -> None:
        # Keep the original, gradient-free update behaviour
        with torch.no_grad():
            super().aggregate_update(X, d, classes)


class TorchLinearClassifier(nn.Module):
    """
    Incremental linear classifier trained with SGD.
    The layer is expanded when new classes arrive.
    """

    def __init__(
        self,
        num_classes: int,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

        self.linear: Optional[nn.Linear] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion = nn.CrossEntropyLoss()

    def _init_or_expand_layer(self, in_features: int) -> None:
        if self.linear is None:
            self.linear = nn.Linear(in_features, self.num_classes).to(self.device)
        elif self.linear.out_features != self.num_classes:
            old = self.linear
            new_layer = nn.Linear(old.in_features, self.num_classes).to(self.device)
            with torch.no_grad():
                new_layer.weight[: old.out_features] = old.weight
                new_layer.bias[: old.out_features] = old.bias
            self.linear = new_layer

        self.optimizer = torch.optim.SGD(
            self.linear.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def add_num_classes(self, num_classes: int) -> None:
        self.num_classes += num_classes
        if self.linear is not None:
            self._init_or_expand_layer(self.linear.in_features)

    def aggregate_update(
        self,
        X: torch.Tensor,
        labels: torch.Tensor,
        classes: Optional[torch.Tensor] = None,
    ) -> None:
        X = X.to(self.device)
        y = torch.argmax(labels, dim=1).to(self.device)

        self._init_or_expand_layer(X.shape[1])

        assert self.linear is not None  # for mypy
        assert self.optimizer is not None

        self.optimizer.zero_grad()
        logits = self.linear(X)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            return torch.zeros((X.size(0), self.num_classes), device=self.device)
        return self.linear(X.to(self.device))


class SklearnLogRegClassifier(nn.Module):
    """
    CPU-based logistic regression using scikit-learn's SGDClassifier with partial_fit.
    Suitable for ablation without changing the surrounding training loop.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.0001,
        lr: float = 0.01,
        device: str = "cuda",
        replay_buffer=None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.lr = lr
        self.device = device

        self._clf: Optional[SGDClassifier] = None
        self._is_fitted = False
        self._needs_init_fit = True
        self.replay_buffer = replay_buffer

    def _ensure_clf(self) -> None:
        if self._clf is None:
            self._clf = SGDClassifier(
                loss="log_loss",
                alpha=self.alpha,
                learning_rate="constant",
                eta0=self.lr,    
                warm_start=True, 
            )

            self._needs_init_fit = True
            self._is_fitted = False

    def add_num_classes(self, num_classes: int) -> None:
        old_num = self.num_classes
        new_num = old_num + num_classes
        self.num_classes = new_num

        # Re-init classifier so the next partial_fit can register the new class list
        self._clf = None
        self._ensure_clf()
        self._needs_init_fit = True
        self._is_fitted = False

        # Bootstrap using replay buffer if available; otherwise next update will do it
        self._bootstrap_from_buffer()

    def aggregate_update(self, X, labels, classes=None):
        self._ensure_clf()
        X_np = X.detach().cpu().numpy()
        y_np = torch.argmax(labels, dim=1).detach().cpu().numpy()

        X_np = X_np / (np.linalg.norm(X_np, axis=1, keepdims=True) + 1e-8)

        if self._needs_init_fit:
            classes_arr = np.arange(self.num_classes)
            self._clf.partial_fit(X_np, y_np, classes=classes_arr)
            self._needs_init_fit = False
            self._is_fitted = True
        else:
            self._clf.alpha = self.alpha * 3.0
            self._clf.partial_fit(X_np, y_np)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self._clf is None or not self._is_fitted:
            return torch.zeros((X.size(0), self.num_classes), device=self.device)

        probs = self._clf.predict_proba(X.detach().cpu().numpy())
        probs_t = torch.from_numpy(probs).to(self.device)
        # Use log probabilities as logits surrogate
        return torch.log(torch.clamp(probs_t, min=1e-8))

    def set_replay_buffer(self, replay_buffer) -> None:
        """
        Attach a replay buffer implementing get_memory_samples(classes) -> (X, y).
        """
        self.replay_buffer = replay_buffer

    def _bootstrap_from_buffer(self) -> None:
        """
        After re-init (e.g., new classes), run a first partial_fit with all classes
        using samples from the replay buffer if available.
        """
        if self.replay_buffer is None:
            return

        class_ids = list(range(self.num_classes))
        embeddings, labels = self.replay_buffer.get_memory_samples(class_ids)

        if embeddings is None or labels is None or embeddings.numel() == 0:
            # No data to bootstrap; keep _needs_init_fit True so next update uses classes=...
            return

        self._ensure_clf()
        X_np = embeddings.detach().cpu().numpy()
        y_np = labels.detach().cpu().numpy()
        self._clf.partial_fit(X_np, y_np, classes=np.arange(self.num_classes))
        self._needs_init_fit = False
        self._is_fitted = True


class NCMClassifier(nn.Module):
    """
    Nearest Class Mean / Prototype classifier.
    Stores running class prototypes and scores samples by negative distance.
    """

    def __init__(self, num_classes: int, device: str = "cuda") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.prototypes: List[Optional[torch.Tensor]] = []
        self.counts: List[int] = []

    def add_num_classes(self, num_classes: int) -> None:
        self.num_classes += num_classes
        self.prototypes.extend([None] * num_classes)
        self.counts.extend([0] * num_classes)

    def aggregate_update(
        self,
        X: torch.Tensor,
        labels: torch.Tensor,
        classes: Optional[torch.Tensor] = None,
    ) -> None:
        if classes is None:
            classes = torch.arange(self.num_classes, device=labels.device)

        X = X.to(self.device)
        y = torch.argmax(labels, dim=1).to(self.device)

        for c in classes:
            class_mask = y == c
            if not torch.any(class_mask):
                continue

            class_feats = X[class_mask]
            new_sum = class_feats.sum(dim=0)
            new_count = class_mask.sum().item()

            if c >= len(self.prototypes):
                self.add_num_classes(int(c.item() + 1 - len(self.prototypes)))

            if self.prototypes[c] is None:
                self.prototypes[c] = class_feats.mean(dim=0)
                self.counts[c] = new_count
            else:
                total_count = self.counts[c] + new_count
                updated_mean = (self.prototypes[c] * self.counts[c] + new_sum) / total_count
                self.prototypes[c] = updated_mean
                self.counts[c] = total_count

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self.prototypes or all(p is None for p in self.prototypes):
            return torch.zeros((X.size(0), self.num_classes), device=self.device)

        valid_protos = [
            p if p is not None else torch.zeros_like(self.prototypes[0])
            for p in self.prototypes
        ]
        proto_tensor = torch.stack(valid_protos, dim=0).to(self.device)
        X = X.to(self.device)
        dists = torch.cdist(X, proto_tensor)
        # Negative distance as a similarity score
        return -dists

