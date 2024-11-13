from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from tqdm import tqdm
from models.samplers.SamplingStrategy import (
    BoundarySampling,
    CentroidSampling,
    EntropySampling,
    HybridSampling,
    KMeansSampling,
    RandomSampling,
    TypicalitySampling,
)
from utils.incremental_data_utils import prepare_data
from models.samplers.MemoryReplayBuffer import MemoryReplayBuffer
from models.rolannet import RolanNET
from scripts.test import evaluate
from utils.utils import split_dataset

sampling_strategies = {
    "centroid": CentroidSampling,
    "entropy": EntropySampling,
    "kmeans": KMeansSampling,
    "random": RandomSampling,
    "typicality": TypicalitySampling,
    "boundary": BoundarySampling,
    "hybrid": HybridSampling,
}


def get_sampling_strategy(strategy_name):
    return sampling_strategies.get(strategy_name.lower(), RandomSampling)


def train_step(
    model: RolanNET,
    inputs: Subset,
    labels: Subset,
    classes: Optional[List[int]],
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[nn.Module] = None,
    calculate_metrics: bool = False,
    total_correct: Optional[int] = None,
    total_samples: Optional[int] = None,
    batch_count: Optional[int] = None,
    running_loss: Optional[float] = None,
) -> Optional[Tuple[int, int, int, int]]:

    labels = process_labels(labels)

    model.update_rolann(inputs, labels, classes=classes)
    outputs = model(inputs)

    if not calculate_metrics:
        return None

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

    return total_correct, total_samples, batch_count, running_loss


def replicate_samples(inputs, labels, desired_size):

    if isinstance(inputs, (list, torch.Tensor)):
        num_samples = len(inputs)
    else:
        raise TypeError("inputs must be a list or torch.Tensor")

    if num_samples == 0:
        raise ValueError("inputs cannot be empty")

    if num_samples != len(labels):
        raise ValueError("inputs and labels must have the same length")

    class_to_inputs = defaultdict(list)
    class_to_labels = defaultdict(list)

    # Group inputs and labels by class
    for i, label in enumerate(labels):
        class_to_inputs[int(label)].append(inputs[i])
        class_to_labels[int(label)].append(label)

    inputs_new = []
    labels_new = []

    for label in class_to_inputs:
        class_inputs = class_to_inputs[label]
        class_labels = class_to_labels[label]
        num_class_samples = len(class_inputs)

        # Calculate repetitions and remainder for this class
        repetitions = desired_size // num_class_samples
        remainder = desired_size % num_class_samples

        # Replicate inputs and labels for this class
        if isinstance(inputs, list):
            replicated_inputs = class_inputs * repetitions + class_inputs[:remainder]
            replicated_labels = class_labels * repetitions + class_labels[:remainder]

        elif isinstance(inputs, torch.Tensor):
            replicated_inputs = torch.cat(
                [torch.stack(class_inputs)] * repetitions
                + [torch.stack(class_inputs)[:remainder]],
                dim=0,
            )
            replicated_labels = torch.cat(
                [torch.stack(class_labels)] * repetitions
                + [torch.stack(class_labels)[:remainder]],
                dim=0,
            )

        # Add the replicated data to the new dataset
        (
            inputs_new.extend(replicated_inputs)
            if isinstance(inputs, list)
            else inputs_new.append(replicated_inputs)
        )
        (
            labels_new.extend(replicated_labels)
            if isinstance(inputs, list)
            else labels_new.append(replicated_labels)
        )

    # If inputs are tensors, concatenate them into a single tensor
    if isinstance(inputs, torch.Tensor):
        inputs_new = torch.cat(inputs_new, dim=0)
        labels_new = torch.cat(labels_new, dim=0)

    return inputs_new, labels_new


def count_samples_per_class(dataloader):
    """
    Count the total number of samples per class in a DataLoader.

    Parameters:
        dataloader (torch.utils.data.DataLoader): DataLoader object containing the dataset.

    Returns:
        class_counts (dict): Dictionary with class labels as keys and sample counts as values.
    """

    class_counts = defaultdict(int)  # Initialize a dictionary with default int (0)

    # Iterate through the DataLoader
    for _, labels in dataloader:
        for label in labels:
            class_counts[int(label)] += 1  # Convert label to int and count

    return dict(class_counts)


def log_samples_per_class(Y):
    """
    Logs the number of samples per class from a given label array Y.

    Parameters:
        Y: numpy array or list of class labels.
    """

    # Count the occurrences of each class
    unique_classes, counts = torch.unique(Y, return_counts=True)

    # Log the results
    for cls, count in zip(unique_classes, counts):
        logger.debug(f"Class {cls}: {count} samples")


def embedding_step(model, embeddings, labels, classes):

    labels = process_labels(labels)
    model.update_rolann(embeddings, labels, classes=classes, is_embedding=True)


def train_ExpansionBuffer(
    model: RolanNET,
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
    classes_per_task = config["incremental"]["classes_per_task"]
    samples_per_task = config["incremental"]["samples_per_task"]
    buffer_batch_size = config["incremental"]["buffer_size"]

    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.Adam(
            model.parameters(), lr=config["model"]["learning_rate"], weight_decay=1e-5
        )
        if model.backbone and not config["model"]["freeze_mode"] == "all"
        else None
    )

    sampling_strategy = get_sampling_strategy(
        config["incremental"]["sampling_strategy"]
    )

    replayBuffer = MemoryReplayBuffer(
        memory_size_per_class=buffer_batch_size,
        classes_per_task=classes_per_task,
        sampling_strategy=sampling_strategy(),
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

    if config["training"]["use_wandb"]:
        import wandb

        wandb.init(project="RolanNet-Model", config=config)
        wandb.watch(model)

    for task in range(config["incremental"]["num_tasks"]):

        logger.info(
            f"Training on classes {task*classes_per_task} to {(task+1)*classes_per_task - 1}"
        )

        current_num_classes = (task + 1) * classes_per_task

        class_range = range(task * classes_per_task, (task + 1) * classes_per_task)

        model.rolann.add_num_classes(classes_per_task)

        # Prepare data for current task
        train_subset = prepare_data(
            train_dataset,
            class_range=class_range,
            samples_per_task=samples_per_task,
        )

        if model.backbone and not config["model"]["freeze_mode"] == "all":
            global_train_loader, val_loader = split_dataset(
                train_subset=train_subset, config=config
            )
        else:
            global_train_loader = DataLoader(
                train_subset, batch_size=config["dataset"]["batch_size"], shuffle=True
            )

        num_epochs = (
            config["training"]["epochs"]
            if not config["model"]["freeze_mode"] == "all"
            else 1
        )

        best_val_loss = float("inf")
        patience = config["training"]["patience"]
        patience_counter = 0

        for epoch in range(num_epochs):

            training_classes = (
                range(0, task * classes_per_task)
                if task != 0
                else range(classes_per_task)
            )

            ## CURRENT TASK TRAINING

            model.train()

            running_loss = 0.0
            total_correct = 0
            total_samples = 0
            batch_count = 0

            logger.debug(f"Training current task in neurons {training_classes}")

            for inputs, labels in tqdm(
                global_train_loader,
                desc=f"Task {task+1} Global Train, Epoch {epoch + 1}",
            ):

                inputs, labels = inputs.to(device), labels.to(device)

                # embeddings = model.backbone(inputs)
                replayBuffer.add_task_samples(inputs, labels, task=task)

                labels = torch.nn.functional.one_hot(
                    labels, num_classes=current_num_classes
                )

                if task == 0:
                    total_correct, total_samples, batch_count, running_loss = (
                        train_step(
                            model,
                            inputs,
                            labels,
                            classes=class_range,
                            criterion=criterion,
                            optimizer=optimizer,
                            total_correct=total_correct,
                            total_samples=total_samples,
                            batch_count=batch_count,
                            running_loss=running_loss,
                            calculate_metrics=True,
                        )
                    )
                else:
                    train_step(
                        model,
                        inputs,
                        labels,
                        classes=training_classes,
                        calculate_metrics=False,
                    )

            if task == 0:
                epoch_loss = running_loss / batch_count
                epoch_acc = (total_correct / total_samples).item()

                logger.info(
                    f"New Task {task+1} Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {100 * epoch_acc}"
                )

                task_train_accuracies[task] = epoch_acc

            ## BUFFER REPLAY TRAINING

            replayBuffer.sample(get_predictions=model.rolann, device=device)

            X_memory, Y_memory = replayBuffer.get_memory_samples(
                classes=range(task * classes_per_task)
            )

            if X_memory.size(0) > 0 and task != 0:

                class_counts = count_samples_per_class(global_train_loader)

                X_replicated, Y_replicated = replicate_samples(
                    X_memory, Y_memory, max(class_counts.values())
                )

                past_task_dataset = TensorDataset(X_replicated, Y_replicated)

                train_subset = prepare_data(
                    train_dataset,
                    class_range=class_range,
                    samples_per_task=samples_per_task,
                )

                concatenated_dataset = ConcatDataset([past_task_dataset, train_subset])
                local_train_loader = DataLoader(
                    concatenated_dataset,
                    batch_size=config["dataset"]["batch_size"],
                    shuffle=True,
                )

                logger.debug(f"Making the local train in the neurons {class_range}")

                running_loss = 0.0
                total_correct = 0
                total_samples = 0
                batch_count = 0

                for inputs, labels in tqdm(
                    local_train_loader,
                    desc=f"Task {task+1} Local Train, Epoch {epoch + 1}",
                ):

                    inputs, labels = inputs.to(device), labels.to(device)

                    labels = torch.nn.functional.one_hot(
                        labels, num_classes=current_num_classes
                    )

                    total_correct, total_samples, batch_count, running_loss = (
                        train_step(
                            model,
                            inputs,
                            labels,
                            criterion=criterion,
                            optimizer=optimizer,
                            total_correct=total_correct,
                            total_samples=total_samples,
                            batch_count=batch_count,
                            running_loss=running_loss,
                            classes=class_range,
                            calculate_metrics=True,
                        )
                    )

                epoch_loss = running_loss / batch_count
                epoch_acc = (total_correct / total_samples).item()

                logger.info(
                    f"New Task {task+1} Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {100 * epoch_acc}"
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
            test_subset = prepare_data(
                test_dataset,
                class_range=range(
                    eval_task * classes_per_task, (eval_task + 1) * classes_per_task
                ),
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

            if (
                eval_task == task and epoch == num_epochs - 1
            ):  # Only log the current task's performance
                if config["training"]["use_wandb"]:
                    wandb.log(
                        {
                            f"train_accuracy_task_{task+1}": epoch_acc,
                            f"train_loss_task_{task+1}": epoch_loss,
                            f"test_accuracy_task_{task+1}": test_accuracy,
                            f"test_loss_task_{task+1}": test_loss,
                        }
                    )

                results["test_loss"].append(test_loss)
                results["test_accuracy"].append(test_accuracy)

        logger.debug(
            f"Memory buffer distribution: {replayBuffer.get_class_distribution()}"
        )

    logger.info(
        f"\nMemory buffer distribution: {replayBuffer.get_class_distribution()}"
    )

    if config["training"]["use_wandb"]:
        wandb.finish()

    return results, task_train_accuracies, task_accuracies


def process_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels * 0.9 + 0.05
