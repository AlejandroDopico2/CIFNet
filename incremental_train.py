import os
from typing import Any, Dict, List
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils.incremental_data_utils import CustomDataset, load_mnist, prepare_data
from utils.utils import split_dataset
from scripts.test import evaluate


def incremental_train(
    model: nn.Module,
    train_dataset: Subset,
    test_dataset: Subset,
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
        optim.Adam(model.parameters(), lr=config["model"]["learning_rate"], weight_decay=1e-5)
        if model.backbone and not config["model"]["freeze_mode"] == "all"
        else None
    )

    results: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    task_accuracies: Dict[int, List[float]] = {
        i: [] for i in range(config["incremental"]["num_tasks"])
    }

    task_train_accuracies: Dict[int, float] = {}

    if config["dataset"]["name"] == "MNIST":

        data_path = os.path.join("Data", "MNIST", "raw")
        flatten = False if model.backbone else True
        X_train, y_train = load_mnist(data_path, kind="train", flatten=flatten)
        X_test, y_test = load_mnist(data_path, kind="t10k", flatten=flatten)

        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()

        y_train = torch.from_numpy(y_train).long()
        y_test = torch.from_numpy(y_test).long()

        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)

    if config["training"]["use_wandb"]:
        import wandb

        wandb.init(project="RolanNet-Model", config=config)
        wandb.watch(model)

    for task in range(config["incremental"]["num_tasks"]):
        classes_per_task = config["incremental"]["classes_per_task"]
        logger.info(
            f"Training on classes {task*classes_per_task} to {(task+1)*classes_per_task - 1}"
        )

        model.rolann.add_num_classes(classes_per_task)

        current_num_classes = (task + 1) * classes_per_task

        class_range = range(task * classes_per_task, (task + 1) * classes_per_task)

        # Prepare data for current task
        train_subset = prepare_data(
            train_dataset,
            class_range=class_range,
            samples_per_task=config["incremental"]["samples_per_task"],
        )

        if model.backbone and not config["model"]["freeze_mode"] == "all":
            train_loader, val_loader = split_dataset(
                train_subset=train_subset, config=config
            )
        else:
            train_loader = DataLoader(
                train_subset, batch_size=config["dataset"]["batch_size"], shuffle=True
            )

        num_epochs = config["training"]["epochs"] if not config["model"]["freeze_mode"] == "all" else 1

        best_val_loss = float("inf")
        patience = config["training"]["patience"]
        patience_counter = 0

        for epoch in range(num_epochs):

            model.train()

            running_loss = 0.0
            total_correct = 0
            total_samples = 0
            batch_count = 0

            for inputs, labels in tqdm(
                train_loader, desc=f"Task {task+1} Epoch {epoch + 1}"
            ):
                inputs, labels = inputs.to(device), labels.to(device)

                labels = torch.nn.functional.one_hot(
                    labels, num_classes=current_num_classes
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

            logger.info(
                f"Task {task+1} Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {100 * epoch_acc:.2f}%"
            )

            task_train_accuracies[task] = epoch_acc

            if model.backbone and not config["model"]["freeze_mode"] == "all":
                val_loss, val_accuracy = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device=device,
                    task=task + 1,
                    mode="Validation",
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break

        # Evaluate on all tasks seen so far
        for eval_task in range(task + 1):

            test_class_range = range(
                eval_task * classes_per_task, (eval_task + 1) * classes_per_task
            )
            test_subset = prepare_data(
                test_dataset,
                class_range=test_class_range,
                samples_per_task=None,
            )

            test_loader = DataLoader(
                test_subset, batch_size=config["dataset"]["batch_size"], shuffle=True
            )

            test_loss, test_accuracy = evaluate(
                model,
                test_loader,
                criterion,
                device=device,
                task=eval_task + 1,
            )

            task_accuracies[eval_task].append(test_accuracy)

            if eval_task == task:  # Only log the current task's performance
                if config["training"]["use_wandb"]:
                    wandb.log(
                        {
                            f"train_accuracy_task_{task+1}": epoch_acc,
                            f"train_loss_task_{task+1}": epoch_loss,
                            f"test_accuracy_task_{task+1}": test_accuracy,
                            f"test_loss_task_{task+1}": test_loss,
                        }
                    )

                results["train_loss"].append(epoch_loss)
                results["train_accuracy"].append(epoch_acc)
                results["test_loss"].append(test_loss)
                results["test_accuracy"].append(test_accuracy)

    # Add task accuracies to results
    for task, accuracies in task_accuracies.items():
        results[f"task_{task+1}_accuracy"] = accuracies

    if config["training"]["use_wandb"]:
        wandb.finish()

    return results, task_train_accuracies, task_accuracies


def process_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels * 0.90 + 0.05
