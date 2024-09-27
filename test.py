from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loguru import logger


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    num_classes: int,
    device: str,
    task: Optional[int] = None,
    mode: str = "Test",
) -> Tuple[float, float]:

    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
            labels = process_labels(labels)

            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))

            pred = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels, dim=1)

            total_correct += (pred == labels).sum()
            total_samples += labels.size(0)

            running_loss += loss.item()
            batch_count += 1

        test_loss = running_loss / batch_count
        test_acc = (total_correct / total_samples).item()

    logger.info(
        f"Task: {task if task else 'N/A'} | {mode} Loss: {test_loss:.4f} | "
        f"{mode} Accuracy: {100 * test_acc:.2f}% "
        f"({total_correct} of {total_samples})"
    )

    return test_loss, test_acc


def process_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels * 0.9 + 0.05
