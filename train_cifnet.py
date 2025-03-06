# Improved and refactored code
from collections import defaultdict
import sys
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from incremental_dataloaders.data_preparation import prepare_data
from models.CIFNet import CIFNet
from models.samplers.MemoryExpansionBuffer import MemoryExpansionBuffer
from models.samplers.SamplingStrategy import (
    BoundarySampling,
    CentroidSampling,
    EntropySampling,
    HybridSampling,
    KMeansSampling,
    RandomSampling,
    TypicalitySampling,
)

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


class MetricTracker:
    """Class to track and compute training/evaluation metrics"""

    def __init__(self):
        self.reset()
        self.history = defaultdict(list)

    def reset(self):
        """Reset batch accumulation counters"""
        self._loss = 0.0
        self._correct = 0
        self._total = 0
        self._batches = 0

    def update(self, loss: float, correct: int, total: int):
        """Update metrics with batch statistics"""
        self._loss += loss
        self._correct += correct
        self._total += total
        self._batches += 1

    @property
    def avg_loss(self) -> float:
        """Compute average loss per batch"""
        return self._loss / self._batches if self._batches > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Compute accuracy percentage"""
        return self._correct / self._total if self._total > 0 else 0.0

    def log_epoch(self, phase: str, task: int):
        """Store current metrics in history and reset counters"""
        self.history[f"{phase}_loss"].append(self.avg_loss)
        self.history[f"{phase}_accuracy"].append(self.accuracy)
        logger.info(
            f"{phase.capitalize()} Task {task} - "
            f"Loss: {self.avg_loss:.4f}, "
            f"Accuracy: {100 * self.accuracy:.2f}%"
        )
        self.reset()

    def get_last_phase_metrics(self, phase: str) -> Dict[str, float]:
        """
        Return the last recorded metrics (loss and accuracy) for a given phase.

        Args:
            phase (str): The phase to retrieve metrics for (e.g., "train" or "test").

        Returns:
            Dict[str, float]: A dictionary containing the last loss and accuracy for the phase.
                              Returns `None` if no metrics are available for the phase.
        """
        loss_key = f"{phase}_loss"
        accuracy_key = f"{phase}_accuracy"

        if loss_key not in self.history or accuracy_key not in self.history:
            logger.warning(f"No metrics found for phase: {phase}")
            return None

        if not self.history[loss_key] or not self.history[accuracy_key]:
            logger.warning(f"No metrics recorded yet for phase: {phase}")
            return None

        last_loss = self.history[loss_key][-1]
        last_accuracy = self.history[accuracy_key][-1]

        return last_loss, last_accuracy


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


class CILTrainer:
    def __init__(self, model: CIFNet, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config["device"]
        self.classes_per_task = config["incremental"]["classes_per_task"]
        self.num_tasks = config["incremental"]["num_tasks"]
        self.current_task = 0
        self.metrics = MetricTracker()

        self._setup_logging()
        self._initialize_components()

    def _setup_logging(self):
        """
        Configure detailed logging with additional context and file logging.
        """
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>",
            level="INFO",
            colorize=True,
        )

    def _initialize_components(self):
        """Initialize training components"""
        self.criterion = nn.CrossEntropyLoss()
        self.expansion_buffer = MemoryExpansionBuffer(
            memory_size_per_class=self.config["incremental"]["buffer_size"],
            classes_per_task=self.classes_per_task,
            sampling_strategy=get_sampling_strategy(
                self.config["incremental"]["sampling_strategy"]
            )(),
        )

    def train_task(
        self, task_id: int, train_dataset: Subset, test_dataset: Subset
    ) -> Tuple[Dict, Dict, Dict]:
        """Train the model on a single task"""

        # Start the task
        self._handle_new_task(task_id, train_dataset)

        # Training phases
        self._train_current_task(task_id)
        self._train_with_buffer(task_id)

        # Evaluation
        train_metrics = self._evaluate(self.train_loader, task_id, mode="Train")
        task_metrics = self._evaluate_tasks(task_id, test_dataset, mode="Test")
        self._log_metrics(task_id, train_metrics, task_metrics)

        return {"train_metrics": train_metrics, "task_metrics": task_metrics}

    def _handle_new_task(self, task: int, train_dataset: Subset):
        """Prepare model and data for new task"""
        logger.info(f"Starting task {task+1}/{self.num_tasks}")
        current_classes = self._get_task_classes(task)

        # Model adjustments
        self.model.rolann.add_num_classes(self.classes_per_task)

        # Data preparation
        self.train_loader = self._prepare_task_data(train_dataset, current_classes)

    def _get_task_classes(self, task: int) -> range:
        """Get class range for current task"""
        return range(task * self.classes_per_task, (task + 1) * self.classes_per_task)

    def _prepare_task_data(self, dataset: Subset, classes: range) -> DataLoader:
        """Prepare data loaders for current task"""
        subset = prepare_data(
            dataset,
            class_range=classes,
            samples_per_task=self.config["incremental"]["samples_per_task"],
        )

        return DataLoader(
            subset,
            batch_size=self.config["dataset"]["batch_size"],
            shuffle=True,
        )

    def _train_current_task(self, task: int) -> Dict[str, float]:
        """Train for a single epoch"""
        self.model.train()

        for inputs, labels in tqdm(
            self.train_loader,
            desc=f"Task {task + 1}",
        ):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = torch.nn.functional.one_hot(
                labels, num_classes=(task + 1) * self.classes_per_task
            )

            # Update model with current task data
            self._train_step(
                inputs=inputs,
                labels=labels,
                task=task,
                classes=None,
                calculate_metrics=False,
                is_embedding=False,
            )

    def _train_with_buffer(self, task: int):
        """Train using the expansion buffer"""
        if task == 0:
            return  # No buffer for the first task

        logger.debug("Training with expansion buffer")
        X_memory, Y_memory = self.expansion_buffer.get_memory_samples(
            classes=range(task * self.classes_per_task)
        )

        if X_memory.size(0) > 0:
            # Replicate samples to balance class distribution
            class_counts = count_samples_per_class(self.train_loader)
            logger.debug(X_memory.size())
            X_replicated, Y_replicated = replicate_samples(
                X_memory, Y_memory, max(class_counts.values())
            )
            past_task_dataset = TensorDataset(X_replicated, Y_replicated)

            replay_loader = DataLoader(
                past_task_dataset,
                batch_size=self.config["dataset"]["batch_size"],
                shuffle=True,
            )

            # Train on replayed data
            self._train_replay(task, replay_loader)

    def _train_replay(self, task: int, replay_loader: DataLoader):
        """Train on replayed data from the buffer"""

        for embeddings, labels in tqdm(replay_loader, desc=f"Task {task + 1} Replay"):
            embeddings, labels = embeddings.to(self.device), labels.to(self.device)
            labels = torch.nn.functional.one_hot(
                labels, num_classes=(task + 1) * self.classes_per_task
            ).float()

            self._train_step(
                inputs=embeddings,
                labels=labels,
                task=task,
                classes=self._get_task_classes(task),
                calculate_metrics=False,
                is_embedding=True,
            )

    def _evaluate_tasks(
        self, task: int, dataset: Subset, mode: str
    ) -> Dict[str, float]:
        """Evaluate the model on all tasks seen so far"""
        metrics = defaultdict(list)

        for eval_task in range(task + 1):
            subset = prepare_data(
                dataset,
                class_range=range(
                    eval_task * self.classes_per_task,
                    (eval_task + 1) * self.classes_per_task,
                ),
                samples_per_task=None,
            )

            loader = DataLoader(
                subset,
                batch_size=self.config["dataset"]["batch_size"],
                shuffle=False,
            )

            loss, accuracy = self._evaluate(loader, eval_task, mode=mode)
            metrics["loss"].append(loss)
            metrics["accuracy"].append(accuracy)

        return metrics

    def _evaluate(
        self, data_loader: DataLoader, task: int, mode: str = "Test"
    ) -> Tuple[float, float]:
        """Evaluate the model on a given data loader"""
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                pred = torch.argmax(outputs, dim=1)
                correct = (pred == labels).sum().item()
                total = labels.size(0)

                self.metrics.update(loss.item(), correct, total)

        # Log and store evaluation metrics
        self.metrics.log_epoch(mode.lower(), task + 1)
        return self.metrics.get_last_phase_metrics(mode.lower())

    def _log_metrics(self, task: int, train_metrics: Dict, task_metrics: Dict):
        """Log metrics to logger and WandB"""
        if self.config["training"]["use_wandb"]:
            log_data = {
                f"train_accuracy_task_{task + 1}": train_metrics["accuracy"],
                f"train_loss_task_{task + 1}": train_metrics["loss"],
                f"test_accuracy_task_{task + 1}": task_metrics["test_accuracy"][-1],
                f"test_loss_task_{task + 1}": task_metrics["test_loss"][-1],
            }

            # Add historical metrics
            for metric, values in self.metrics.history.items():
                log_data[metric] = values[-1] if values else 0.0

            self.wandb.log(log_data)

    def _train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        task: int,
        classes: List[int],
        calculate_metrics: bool = False,
        is_embedding: bool = False,
    ) -> Optional[Tuple[float, int, int]]:
        """
        Core training step handling both model updates and optional metric calculation

        Args:
            inputs: Batch of input tensors
            labels: Ground truth labels
            task: Current task ID
            classes: List of active classes for this task
            calculate_metrics: Whether to compute loss/accuracy
            is_embedding: Whether inputs are precomputed embeddings

        Returns:
            Tuple of (loss, correct, total) if calculate_metrics=True, else None
        """
        # Process labels with smoothing
        processed_labels = self._process_labels(labels)

        # Update memory buffer with current samples
        if not is_embedding:
            with torch.no_grad():
                embeddings = self.model.backbone(inputs)

            self.expansion_buffer.add_task_samples(
                embeddings, processed_labels.detach(), task=task
            )

        # Update ROLANN layer
        self.model.update_rolann(
            inputs.detach(),
            processed_labels,
            classes=classes,
            is_embedding=is_embedding,
        )

        if not calculate_metrics or is_embedding:
            return None

        # Forward pass
        outputs = self.model(inputs)

        # Calculate loss
        loss = self.criterion(outputs, torch.argmax(processed_labels, dim=1))

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        true_labels = torch.argmax(processed_labels, dim=1)
        correct = (preds == true_labels).sum().item()
        total = true_labels.size(0)

        return loss.item(), correct, total

    def _process_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to ground truth labels"""
        return labels * 0.9 + 0.05
