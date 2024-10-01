from typing import Any, Dict, List, Union
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import wandb
from incremental_data_utils import prepare_data
from test import evaluate


def train(
    model: nn.Module,
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: Dict[str, Any],
) -> Dict[str, List[float]]:

    logger.remove()  # Remove default logger to customize it
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    
    device = config["device"]
    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
        if model.backbone and not config["freeze_mode"] == "all"
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

    num_train = min(config["num_instances"], len(train_dataset)) if config["num_instances"] is not None else len(train_dataset)
    train_indices = torch.randperm(len(train_dataset))[:num_train]
    train_subset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(
        train_subset, batch_size=config["batch_size"], shuffle=True
    )

    task_accuracies: Dict[int, List[float]] = {
        i: [] for i in range(config["num_tasks"])
    }

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

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels, dim=1)
            total_correct += (pred == labels).sum()
            total_samples += labels.size(0)

            running_loss += loss.item()
            batch_count += 1

        epoch_loss = running_loss / batch_count
        epoch_acc = (total_correct / total_samples).item()

        logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {100 * epoch_acc}")

        # test_loss, test_accuracy = evaluate(
        #     model, test_loader, criterion, epoch, config["num_classes"], device
        # )

        # if config["use_wandb"]:
        #     wandb.log(
        #         {
        #             "train_accuracy": epoch_acc,
        #             "train_loss": epoch_loss,
        #             "test_accuracy": test_accuracy,
        #             "test_loss": test_loss,
        #         }
        #     )

        # results["train_loss"].append(epoch_loss)
        # results["train_accuracy"].append(epoch_acc)
        # results["test_loss"].append(test_loss)
        # results["test_accuracy"].append(test_accuracy)

    for eval_task in range(config["num_tasks"]):
        test_subset = prepare_data(
            test_dataset,
            class_range=range(eval_task * config["classes_per_task"], (eval_task + 1) * config["classes_per_task"]),
        )

        test_loader = DataLoader(
            test_subset, batch_size=config["batch_size"], shuffle=True
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            epoch,
            num_classes=(eval_task + 1) * config["classes_per_task"],
            device=device,
            task=eval_task + 1,
        )

        task_accuracies[eval_task].append(test_accuracy)

    if config["use_wandb"]:
        wandb.finish()

    return results, task_accuracies


def process_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels * 0.9 + 0.05
