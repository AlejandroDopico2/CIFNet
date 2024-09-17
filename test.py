from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    num_classes: int,
    device: str,
) -> Tuple[float, float]:

    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if num_classes == 2:
                labels = labels.float()
            else:
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

    print(
        f"Epoch {epoch + 1}, Test Loss: {test_loss}, Test Accuracy: {100 * test_acc} ({total_correct} of {total_samples})"
    )

    return test_loss, test_acc


def process_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels * 0.9 + 0.05
