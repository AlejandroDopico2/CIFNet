# Improved and refactored code
from collections import defaultdict
import sys
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
import wandb
import numpy as np

from incremental_dataloaders.data_preparation import prepare_data
from models.CIFNet import CIFNet
from models.replay.replay_store import create_replay_store
from models.samplers.SamplingStrategy import (
    BoundarySampling,
    CentroidSampling,
    EntropySampling,
    HybridSampling,
    KMeansSampling,
    RandomSampling,
    TypicalitySampling,
    HerdingSampling,
    PrototypeSampling,
    GaussianPrototypeSampling,
)

sampling_strategies = {
    "centroid": CentroidSampling,
    "entropy": EntropySampling,
    "kmeans": KMeansSampling,
    "random": RandomSampling,
    "typicality": TypicalitySampling,
    "boundary": BoundarySampling,
    "hybrid": HybridSampling,
    "herding": HerdingSampling,
    "prototype": PrototypeSampling,
    "gaussian_prototype": GaussianPrototypeSampling,
}


def get_sampling_strategy(strategy_name: str, **kwargs):
    """
    Return an initialized sampling strategy.

    `kwargs` are forwarded to the sampler's `__init__`, e.g.:
        get_sampling_strategy("prototype", n_prototypes=10)
    """
    strategy_cls = sampling_strategies.get(strategy_name.lower(), RandomSampling)
    return strategy_cls(**kwargs) if kwargs else strategy_cls()


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
        sampler_name = self.config["incremental"]["sampling_strategy"]
        sampler_kwargs = self.config["incremental"].get("sampling_strategy_kwargs", {})

        self.replay_store = create_replay_store(
            self.config,
            sampling_strategy=get_sampling_strategy(
                sampler_name,
                **sampler_kwargs,
            ),
            device=self.device,
        )
        # Backwards-compatible alias
        self.expansion_buffer = self.replay_store

        if hasattr(self.model.classifier, "set_replay_buffer"):
            self.model.classifier.set_replay_buffer(self.replay_store)

    def train_task(
        self, task_id: int, train_dataset: Subset, test_dataset: Subset
    ) -> Dict[str, List[float]]:
        """Train the model on a single task"""

        # Start the task
        self._handle_new_task(task_id, train_dataset)

        # Training phases
        if self.model.classifier_type == "rolann":
            # Single backbone pass → embedding dataset → ROLANN training.
            curr_emb, curr_lbl = self._extract_all_embeddings(task_id)
            replay_emb, replay_lbl = self._get_old_class_replay_memory(
                task_id,
                samples_per_class=self._per_class_max(curr_lbl),
            )
            self._train_rolann_from_embeddings(
                task_id, curr_emb, curr_lbl, replay_emb, replay_lbl
            )
        else:
            self._train_joint_task_and_buffer(task_id)

        self.replay_store.maintain()

        # Optional: measure CVAE generation quality against the reference buffer.
        # Enabled via config key ``cvae_measure_quality: true``.
        self._maybe_measure_cvae_quality(task_id)

        # Evaluation
        test_metrics = self._evaluate_tasks(task_id, test_dataset, mode="Test")

        return test_metrics

    def _handle_new_task(self, task: int, train_dataset: Subset):
        """Prepare model and data for new task"""
        logger.info(f"🚀 Starting task {task+1}/{self.num_tasks}")
        current_classes = self._get_task_classes(task)

        # Model adjustments
        self.model.add_num_classes(self.classes_per_task)

        if hasattr(self.model.classifier, "set_replay_buffer"):
            self.model.classifier.set_replay_buffer(self.replay_store)

        # Data preparation
        self.train_loader = self._prepare_task_data(train_dataset, current_classes)

    def _get_task_classes(self, task: int) -> range:
        """Get class range for current task"""
        return range(task * self.classes_per_task, (task + 1) * self.classes_per_task)

    def _current_task_samples_per_class(self) -> int:
        """Samples per class in the current task stream (replay target count)."""
        counts = count_samples_per_class(self.train_loader)
        if not counts:
            return 0
        return max(counts.values())

    def _per_class_max(self, labels: torch.Tensor) -> int:
        """Max samples per class in a flat index-label tensor."""
        if labels.numel() == 0:
            return 0
        counts: Dict[int, int] = defaultdict(int)
        for c in labels.view(-1).tolist():
            counts[int(c)] += 1
        return max(counts.values()) if counts else 0

    def _get_old_class_replay_memory(
        self,
        task: int,
        samples_per_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Old-class embeddings balanced to current-task per-class count.

        *samples_per_class* overrides the DataLoader-based count when the
        caller already has the current-task label tensor available (avoids
        a second pass through the DataLoader).
        """
        if task == 0:
            return (
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device, dtype=torch.long),
            )

        old_classes = list(range(task * self.classes_per_task))
        per_class = (
            samples_per_class
            if samples_per_class is not None
            else self._current_task_samples_per_class()
        )
        kwargs: Dict[str, Any] = {}
        if per_class > 0:
            kwargs["samples_per_class"] = per_class

        return self.replay_store.get_memory_samples(old_classes, **kwargs)

    def _maybe_measure_cvae_quality(self, task_id: int) -> None:
        """
        Measure CVAE generation quality against the measurement-only reference
        buffer and log the results.

        Controlled by two config keys:
        - ``cvae_measure_quality: true``   — enables the measurement (default false)
        - ``ref_buffer_per_class: N``      — must be > 0 for the reference buffer
          to exist; set this in the store config (default 0 = disabled)
        - ``cvae_quality_n_gen: N``        — samples per class to generate for
          comparison (default 64)
        """
        inc_cfg = self.config.get("incremental", {})
        if not inc_cfg.get("cvae_measure_quality", False):
            return

        store = self.replay_store
        if not hasattr(store, "measure_generation_quality"):
            return
        if not getattr(store, "_ref_buffer", {}):
            logger.warning(
                "cvae_measure_quality=true but ref_buffer_per_class=0 — "
                "no reference data collected. Set ref_buffer_per_class > 0."
            )
            return

        n_gen = int(inc_cfg.get("cvae_quality_n_gen", 64))
        current_task_classes = list(self._get_task_classes(task_id))

        report = store.measure_generation_quality(n_gen_per_class=n_gen)
        if not report or not report.get("agg"):
            return

        from models.replay.cvae_metrics import format_generation_quality_table
        table = format_generation_quality_table(
            report,
            task_id=task_id,
            current_task_classes=current_task_classes,
        )
        logger.info(table)

        # Persist the report in the store for downstream access (e.g. wandb).
        store.last_generation_quality_report = report

        agg = report["agg"]
        if wandb.run is not None:
            wandb.log(
                {
                    "cvae/gen_mean_centroid_l2": agg["mean_centroid_l2"],
                    "cvae/gen_mean_std_ratio": agg["mean_std_ratio"],
                    "cvae/gen_mean_cosine_sim": agg["mean_cosine_sim"],
                    "cvae/gen_mean_fd": agg["mean_fd"],
                    "cvae/gen_mean_mmd": agg["mean_mmd"],
                    "cvae/gen_worst_fd": agg["worst_class_fd"],
                    "cvae/gen_worst_mmd": agg["worst_class_mmd"],
                    "task": task_id + 1,
                },
                step=task_id,
            )

    def _log_replay_debug(self, task: int, phase: str, **kwargs) -> None:
        """Lightweight replay / calibration logging (enable via config)."""
        if not self.config.get("incremental", {}).get("replay_debug", True):
            return
        parts = [f"[replay task {task + 1} | {phase}]"]
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                parts.append(f"{key}=shape{tuple(val.shape)}")
            elif isinstance(val, dict):
                parts.append(f"{key}={val}")
            else:
                parts.append(f"{key}={val}")
        logger.info(" ".join(parts))

    def _calibration_labels_for_new_neurons(
        self, batch_size: int, task: int
    ) -> torch.Tensor:
        """
        Targets for replay: keep new neurons inactive on old-class embeddings.
        Only columns for current-task classes are updated (see classes= in replay).
        """
        num_classes = (task + 1) * self.classes_per_task
        labels = torch.zeros(batch_size, num_classes, device=self.device)
        return self._process_labels(labels)

    def _summarize_per_class_counts(
        self, labels: torch.Tensor
    ) -> Dict[int, int]:
        labels = labels.view(-1).long().cpu()
        counts: Dict[int, int] = {}
        for c in labels.unique().tolist():
            counts[int(c)] = int((labels == c).sum().item())
        return counts

    def _prepare_task_data(self, dataset: Subset, classes: range) -> DataLoader:
        """Prepare data loaders for current task"""
        subset = prepare_data(
            dataset,
            class_range=classes,
            samples_per_task=None,
        )

        return DataLoader(
            subset,
            batch_size=self.config["dataset"]["batch_size"],
            shuffle=True,
        )

    @torch.no_grad()
    def _extract_all_embeddings(
        self, task: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single backbone pass over the current task's training loader.

        Returns CPU tensors ``(embeddings, label_indices)`` covering the entire
        training set.  As a side-effect, every batch is staged in the replay
        store (``add_task_samples``) so the CVAE / buffer receives all real
        embeddings for this task — no second pass is needed.
        """
        self.model.eval()
        emb_list: List[torch.Tensor] = []
        lbl_list: List[torch.Tensor] = []

        for inputs, labels in tqdm(
            self.train_loader,
            desc=f"Extracting task {task + 1}",
            leave=False,
        ):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            emb = self.model.backbone(inputs)
            # Stage for CVAE / replay buffer (done exactly once per sample).
            self.replay_store.add_task_samples(emb.detach(), labels.detach())
            emb_list.append(emb.cpu())
            lbl_list.append(labels.cpu())

        if not emb_list:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        return torch.cat(emb_list, dim=0), torch.cat(lbl_list, dim=0)

    def _train_rolann_from_embeddings(
        self,
        task: int,
        curr_emb: torch.Tensor,
        curr_lbl: torch.Tensor,
        replay_emb: torch.Tensor,
        replay_lbl: torch.Tensor,
    ) -> None:
        """
        Train the ROLANN classifier on pre-extracted embeddings.

        No backbone call is made here; all inputs are already in embedding
        space.  Two sequential phases mirror the previous online flow:

        **Phase 1 — replay calibration** (skipped at task 0)
            Old-class embeddings (from CVAE or buffer) are passed with
            calibration labels (zeros for new neurons) so that the freshly
            added output nodes learn to stay inactive on old-class data.

        **Phase 2 — current-task learning**
            Current-task embeddings are passed with one-hot labels so that
            the new output nodes learn to activate on the new classes.
        """
        batch_size = self.config["dataset"]["batch_size"]
        new_classes = list(self._get_task_classes(task))
        num_classes = (task + 1) * self.classes_per_task

        # ── Phase 1: replay calibration ────────────────────────────────────
        if replay_emb.numel() > 0:
            source_summary = {}
            if hasattr(self.replay_store, "last_replay_sources"):
                source_summary = self.replay_store.last_replay_sources
            self._log_replay_debug(
                task,
                "replay_phase",
                n_samples=int(replay_emb.size(0)),
                replay_per_class=self._summarize_per_class_counts(replay_lbl),
                new_neurons=new_classes,
                replay_sources=source_summary,
            )

            replay_loader = DataLoader(
                TensorDataset(replay_emb),
                batch_size=batch_size,
                shuffle=True,
            )
            for (emb_b,) in tqdm(
                replay_loader, desc=f"Replay {task + 1}", leave=False
            ):
                emb_b = emb_b.to(self.device)
                cal_labels = self._calibration_labels_for_new_neurons(
                    emb_b.size(0), task
                )
                self.model.update_classifier(
                    emb_b, cal_labels, classes=new_classes, is_embedding=True
                )
        elif task > 0:
            self._log_replay_debug(task, "skip", reason="empty replay memory")

        # ── Phase 2: current-task learning ─────────────────────────────────
        self._log_replay_debug(
            task,
            "current_task",
            update_classes=new_classes,
            n_samples=int(curr_emb.size(0)),
        )
        curr_lbl_oh = torch.nn.functional.one_hot(
            curr_lbl.long(), num_classes=num_classes
        ).float()
        curr_loader = DataLoader(
            TensorDataset(curr_emb, curr_lbl_oh),
            batch_size=batch_size,
            shuffle=True,
        )
        for emb_b, lbl_b in tqdm(
            curr_loader, desc=f"Task {task + 1}", leave=False
        ):
            emb_b = emb_b.to(self.device)
            lbl_b = lbl_b.to(self.device)
            self.model.update_classifier(
                emb_b,
                self._process_labels(lbl_b),
                classes=None,
                is_embedding=True,
            )

    def _train_joint_task_and_buffer(self, task: int) -> None:
        """
        For non-ROLANN classifiers: extract all embeddings once, concatenate
        with replay embeddings, shuffle, and train in one joint pass.
        """
        class_count = (task + 1) * self.classes_per_task

        # Single backbone pass — stages embeddings in the replay store too.
        curr_emb, curr_lbl_idx = self._extract_all_embeddings(task)
        if curr_emb.numel() == 0:
            return

        curr_lbl_oh = torch.nn.functional.one_hot(
            curr_lbl_idx.long(), num_classes=class_count
        ).float().to(self.device)
        curr_emb = curr_emb.to(self.device)

        # Old-class replay (buffer or CVAE-generated).
        replay_emb, replay_lbl = self._get_old_class_replay_memory(
            task, samples_per_class=self._per_class_max(curr_lbl_idx)
        )

        if replay_emb.numel() > 0:
            replay_lbl_oh = torch.nn.functional.one_hot(
                replay_lbl.long().to(self.device), num_classes=class_count
            ).float()
            all_emb = torch.cat([curr_emb, replay_emb.to(self.device)], dim=0)
            all_lbl = torch.cat([curr_lbl_oh, replay_lbl_oh], dim=0)
        else:
            all_emb = curr_emb
            all_lbl = curr_lbl_oh

        # Shuffle jointly before training.
        perm = torch.randperm(all_emb.size(0), device=self.device)
        all_emb = all_emb[perm]
        all_lbl = all_lbl[perm]

        joint_loader = DataLoader(
            TensorDataset(all_emb, all_lbl),
            batch_size=self.config["dataset"]["batch_size"],
            shuffle=False,
        )
        for emb_b, lbl_b in tqdm(
            joint_loader, desc=f"Joint task {task + 1}", leave=False
        ):
            self._train_step(
                inputs=emb_b,
                labels=lbl_b,
                task=task,
                classes=None,
                calculate_metrics=False,
                is_embedding=True,
            )

    def _evaluate_tasks(
        self, task: int, dataset: Subset, mode: str
    ) -> Dict[str, List[float]]:
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

        logger.info(f"Task {task + 1} - Mean Accuracy: {np.mean(metrics['accuracy']) * 100:.2f}%")

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
                f"train_accuracy_task_{task + 1}": train_metrics[1] * 100,
                f"train_loss_task_{task + 1}": train_metrics[0],
                f"test_accuracy_task_{task + 1}": task_metrics["accuracy"][-1] * 100,
                f"test_loss_task_{task + 1}": task_metrics["loss"][-1],
            }

            # Add historical metrics
            for metric, values in self.metrics.history.items():
                log_data[metric] = values[-1] if values else 0.0

            wandb.log(log_data)

    def _train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        task: int,
        classes: Optional[List[int]],
        calculate_metrics: bool = False,
        is_embedding: bool = True,
    ) -> Optional[Tuple[float, int, int]]:
        """
        Single classifier update on a batch of **pre-extracted embeddings**.

        ``is_embedding`` is ``True`` by default — all callers now pass
        embeddings directly.  Backbone extraction and replay-store staging are
        handled upstream in ``_extract_all_embeddings``.
        """
        processed_labels = self._process_labels(labels)

        self.model.update_classifier(
            inputs.detach(),
            processed_labels,
            classes=classes,
            is_embedding=is_embedding,
        )

        if not calculate_metrics:
            return None

        outputs = self.model.classifier(inputs.detach())
        loss = self.criterion(outputs, torch.argmax(processed_labels, dim=1))
        preds = torch.argmax(outputs, dim=1)
        true_labels = torch.argmax(processed_labels, dim=1)
        correct = (preds == true_labels).sum().item()
        total = true_labels.size(0)
        return loss.item(), correct, total

    def _process_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to ground truth labels"""
        return labels * 0.95 + 0.025
