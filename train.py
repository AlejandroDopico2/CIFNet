from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from test import test


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
) -> Dict[str, List[float]]:

    device = config["device"]
    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
        if model.backbone or not config["freeze"]
        else None
    )

    results: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    if config["use_wandb"]:
        wandb.init(project="RolanNet-Model", config=config)
        wandb.watch(model)

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = 0

        if config["reset"]:
            model.reset_rolann()

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs, labels = inputs.to(device), labels.to(device)

            if config["num_classes"] == 2:
                labels = labels.float()
            else:
                labels = torch.nn.functional.one_hot(
                    labels, num_classes=config["num_classes"]
                )

            labels = process_labels(labels)

            model.update_rolann(inputs, labels)
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))

            # if optimizer:
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels, dim=1)
            total_correct += (pred == labels).sum()
            total_samples += labels.size(0)

            running_loss += loss.item()
            batch_count += 1

        epoch_loss = running_loss / batch_count
        epoch_acc = (total_correct / total_samples).item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {100 * epoch_acc}")

        test_loss, test_accuracy = test(
            model, test_loader, criterion, epoch, config["num_classes"], device
        )

        if config["use_wandb"]:
            wandb.log(
                {
                    "train_accuracy": epoch_acc,
                    "train_loss": epoch_loss,
                    "test_accuracy": test_accuracy,
                    "test_loss": test_loss,
                }
            )

        results["train_loss"].append(epoch_loss)
        results["train_accuracy"].append(epoch_acc)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)

    if config["use_wandb"]:
        wandb.finish()

    return results


def process_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels * 0.9 + 0.05
