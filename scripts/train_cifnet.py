# Improved and refactored code
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from incremental_dataloaders.data_preparation import prepare_data
from models.CIFNet import CIFNet
from models.samplers.MemoryExpansionBuffer import MemoryExpansionBuffer
from scripts.experience_replay_incremental_train import get_sampling_strategy
from utils.utils import split_dataset

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
        self.metrics = MetricTracker()

        
        self._setup_logging()
        self._initialize_components()

    def _setup_logging(self):
        """Configure logging and experiment tracking"""
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )

        if self.config["training"]["use_wandb"]:
            import wandb
            self.wandb = wandb
            self.wandb.init(project="RolanNet-Model", config=self.config)
            self.wandb.watch(self.model)

    def _initialize_components(self):
        """Initialize training components"""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()
        self.expansion_buffer = MemoryExpansionBuffer(
            memory_size_per_class=self.config["incremental"]["buffer_size"],
            classes_per_task=self.classes_per_task,
            sampling_strategy=get_sampling_strategy(
                self.config["incremental"]["sampling_strategy"]
            )(),
        )

    def _create_optimizer(self):
        """Conditionally create optimizer based on model configuration"""
        if self.model.backbone and not self.config["model"]["freeze_mode"] == "all":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config["model"]["learning_rate"],
                weight_decay=1e-5
            )
        return None

    def train(self, train_dataset: Subset, test_dataset: Subset) -> Tuple[Dict, Dict, Dict]:
        """Main training loop"""
        results = defaultdict(list)
        task_accuracies = {i: [] for i in range(self.num_tasks)}
        task_train_accuracies = {}

        for task in range(self.num_tasks):
            self._handle_new_task(task, train_dataset)
            
            # Training phases
            train_metrics = self._train_task(task, train_dataset)
            self._train_with_buffer(task, train_dataset)
            
            # Evaluation
            task_metrics = self._evaluate_tasks(task, test_dataset)
            self._log_metrics(task, train_metrics, task_metrics)
            
            # Update results
            task_train_accuracies[task] = train_metrics["accuracy"]
            results.update({
                "test_loss": task_metrics["test_loss"],
                "test_accuracy": task_metrics["test_accuracy"]
            })

        if self.config["training"]["use_wandb"]:
            self.wandb.finish()

        return results, task_train_accuracies, task_accuracies

    def _handle_new_task(self, task: int, train_dataset: Subset):
        """Prepare model and data for new task"""
        logger.info(f"Starting task {task+1}/{self.num_tasks}")
        current_classes = self._get_task_classes(task)
        
        # Model adjustments
        self.model.rolann.add_num_classes(self.classes_per_task)
        
        # Data preparation
        self.train_loader, self.val_loader = self._prepare_task_data(
            train_dataset, current_classes
        )

    def _get_task_classes(self, task: int) -> range:
        """Get class range for current task"""
        return range(
            task * self.classes_per_task,
            (task + 1) * self.classes_per_task
        )

    def _prepare_task_data(self, dataset: Subset, classes: range) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for current task"""
        subset = prepare_data(
            dataset,
            class_range=classes,
            samples_per_task=self.config["incremental"]["samples_per_task"],
        )

        if self.model.backbone and not self.config["model"]["freeze_mode"] == "all":
            return split_dataset(subset, self.config)
            
        return (
            DataLoader(
                subset,
                batch_size=self.config["dataset"]["batch_size"],
                shuffle=True,
            ),
            None
        )
    
    def _train_task(self, task: int, train_dataset: Subset) -> Dict[str, float]:
        """Train the model on the current task"""
        logger.debug(f"Training on task {task + 1}")

        train_metrics = {
            "loss": [],
            "accuracy": [],
        }

        epoch_metrics = self._train_epoch(task)
        train_metrics["loss"].append(epoch_metrics["loss"])
        train_metrics["accuracy"].append(epoch_metrics["accuracy"])

        return {
            "loss": train_metrics["loss"][-1],
            "accuracy": train_metrics["accuracy"][-1],
        }

    def _train_epoch(self, task: int) -> Dict[str, float]:
        """Train for a single epoch"""
        self.model.train()
        self.metrics.reset()

        for inputs, labels in tqdm(
            self.train_loader,
            desc=f"Task {task + 1}",
        ):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = torch.nn.functional.one_hot(labels, num_classes=(task + 1) * self.classes_per_task)

            # Update model with current task data
            step_result = self._train_step(
                inputs=inputs,
                labels=labels,
                task=task,
                classes=self._get_task_classes(task),
                calculate_metrics=True,
                is_embedding=False
            )

            if step_result:
                loss, correct, total = step_result
                self.metrics.update(loss, correct, total)

        self.metrics.log_epoch("train", task + 1)
        return {
            "loss": self.metrics.avg_loss,
            "accuracy": self.metrics.accuracy
        }

    def _train_with_buffer(self, task: int, train_dataset: Subset):
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
                X_memory, Y_memory, max(class_counts.values()))
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
        self.metrics.reset()

        for embeddings, labels in tqdm(replay_loader, desc=f"Task {task + 1} Replay"):
            logger.debug(embeddings.size())
            embeddings, labels = embeddings.to(self.device), labels.to(self.device)
            labels = torch.nn.functional.one_hot(
                labels, 
                num_classes=(task + 1) * self.classes_per_task
            ).float()

            step_result = self._train_step(
                inputs=embeddings,
                labels=labels,
                task=task,
                classes=self._get_task_classes(task),
                calculate_metrics=True,
                is_embedding=True
            )

            if step_result:
                loss, correct, total = step_result
                self.metrics.update(loss, correct, total)

        # Log and store replay metrics
        self.metrics.log_epoch("replay", task + 1)

    def _evaluate_tasks(self, task: int, test_dataset: Subset) -> Dict[str, float]:
        """Evaluate the model on all tasks seen so far"""
        test_metrics = {
            "test_loss": [],
            "test_accuracy": [],
        }

        for eval_task in range(task + 1):
            test_subset = prepare_data(
                test_dataset,
                class_range=range(
                    eval_task * self.classes_per_task,
                    (eval_task + 1) * self.classes_per_task,
                ),
                samples_per_task=None,
            )

            test_loader = DataLoader(
                test_subset,
                batch_size=self.config["dataset"]["batch_size"],
                shuffle=True,
            )

            test_loss, test_accuracy = self._evaluate(test_loader, eval_task)
            test_metrics["test_loss"].append(test_loss)
            test_metrics["test_accuracy"].append(test_accuracy)

        return test_metrics

    def _evaluate(self, data_loader: DataLoader, task: int, mode: str = "Test") -> Tuple[float, float]:
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
        return self.metrics.avg_loss, self.metrics.accuracy

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
        is_embedding: bool = False
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
        if self.expansion_buffer:
            self.expansion_buffer.add_task_samples(
                inputs.detach(),
                processed_labels.detach(),
                task=task
            )

        # Update ROLANN layer
        self.model.update_rolann(
            inputs.detach(),
            processed_labels,
            classes=classes,
            is_embedding=is_embedding
        )

        if not calculate_metrics:
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

    