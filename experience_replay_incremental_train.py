from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from incremental_data_utils import prepare_data
from models.MemoryReplayBuffer import MemoryReplayBuffer
from test import evaluate


def trainER(
    model: nn.Module,
    train_dataset: Subset,
    test_dataset: Subset,
    config: Dict[str, Any],
) -> Dict[str, List[float]]:

    device = config["device"]
    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
        if model.backbone and not config["freeze"]
        else None
    )

    # buffer_size = 100
    buffer_batch_size = 64
    replayBuffer = MemoryReplayBuffer(config["buffer_size"])

    results: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    task_accuracies: Dict[int, List[float]] = {
        i: [] for i in range(config["num_tasks"])
    }

    if config["use_wandb"]:
        import wandb

        wandb.init(project="RolanNet-Model", config=config)
        wandb.watch(model)

    for task in range(config["num_tasks"]):
        classes_per_task = config["classes_per_task"]
        print(
            f"\nTraining on classes {task*classes_per_task} to {(task+1)*classes_per_task - 1}"
        )

        current_num_classes = (task + 1) * classes_per_task

        model.rolann.add_num_classes(classes_per_task)

        # Prepare data for current task
        train_subset = prepare_data(
            train_dataset,
            class_range=range(task * classes_per_task, (task + 1) * classes_per_task),
            samples_per_task=config["samples_per_task"],
        )

        train_loader = DataLoader(
            train_subset, batch_size=config["batch_size"], shuffle=True
        )

        num_epochs = config["epochs"] if not config["freeze"] else 1

        for epoch in range(num_epochs):

            model.train()

            running_loss = 0.0
            total_correct = 0
            total_samples = 0
            batch_count = 0

            if config["reset"]:
                model.reset_rolann()

            for inputs, labels in tqdm(
                train_loader, desc=f"Task {task+1} Epoch {epoch + 1}"
            ):
                inputs, labels = inputs.to(device), labels.to(device)

                labels = torch.nn.functional.one_hot(
                    labels, num_classes=current_num_classes
                )

                x_memory, y_memory = replayBuffer.get_memory_samples(
                    batch_size=buffer_batch_size, num_classes=current_num_classes
                )

                x_combined = (
                    torch.cat([x_memory, inputs], dim=0)
                    if x_memory.size(0) > 0
                    else inputs
                )
                y_combined = (
                    torch.cat([y_memory, labels], dim=0)
                    if y_memory.size(0) > 0
                    else labels
                )

                y_combined = process_labels(y_combined)

                model.update_rolann(x_combined, y_combined)
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

                replayBuffer.add_samples(inputs, labels)
            

            epoch_loss = running_loss / batch_count
            epoch_acc = (total_correct / total_samples).item()

            print(
                f"Task {task+1} Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {100 * epoch_acc}"
            )

            # Evaluate on all tasks seen so far
            for eval_task in range(task + 1):
                test_subset = prepare_data(
                    test_dataset,
                    class_range=range(
                        eval_task * classes_per_task, (eval_task + 1) * classes_per_task
                    ),
                    samples_per_task=config["samples_per_task"],
                )

                test_loader = DataLoader(
                    test_subset, batch_size=config["batch_size"], shuffle=True
                )

                test_loss, test_accuracy = evaluate(
                    model,
                    test_loader,
                    criterion,
                    epoch,
                    num_classes=current_num_classes,
                    device=device,
                    task=eval_task + 1,
                )

                task_accuracies[eval_task].append(test_accuracy)

                if eval_task == task:  # Only log the current task's performance
                    if config["use_wandb"]:
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

    print(f"\nMemory buffer distribution: {replayBuffer.get_class_distribution()}")

    if config["use_wandb"]:
        wandb.finish()

    return results, task_accuracies


def process_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels * 0.9 + 0.05
