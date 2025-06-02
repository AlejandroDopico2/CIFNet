import argparse
import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional
import numpy as np
import yaml
from loguru import logger
from codecarbon import EmissionsTracker
from torch.utils.data import Subset

from incremental_dataloaders.data_preparation import get_dataset_instance
from train_cifnet import CILTrainer
from utils.model_utils import build_incremental_model
from utils.plotting import (
    plot_mean_accuracy_progression,
    calculate_cl_metrics,
    plot_task_progression,
)


class ExperimentRunner:
    """Main experiment orchestration class"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._validate_config()
        self._init_paths()
        self.model = None
        self.trainer = None

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load and return configuration file"""
        if config_path is None:
            args = self._parse_args()
            config_path = args.config_path

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="Train an incremental CV model with specified parameters"
        )
        parser.add_argument(
            "-p",
            "--config_path",
            type=str,
            required=True,
            help="Path to YAML configuration file",
        )
        return parser.parse_args()

    def _setup_logging(self):
        """Configure logging system"""
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> - <level>{message}</level>",
        )

    def _validate_config(self):
        """Validate configuration structure"""
        required_keys = {
            "device",
            "dataset",
            "model",
            "training",
            "incremental",
            "output_dir",
            "rolann",
        }
        if not required_keys.issubset(self.config.keys()):
            missing = required_keys - self.config.keys()
            raise ValueError(f"Missing config keys: {missing}")

    def _init_paths(self):
        """Initialize output directories"""
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.emissions_dir = self.output_dir / "emissions"
        self.emissions_dir.mkdir(exist_ok=True)

    def _log_config(self):
        """Log configuration parameters"""
        logger.info("=" * 50)
        logger.info(f"Device: {self.config['device']}")
        logger.info(f"Dataset: {self.config['dataset']['name']}")
        logger.info(f"Backbone: {self.config['model']['backbone']}")
        logger.info("=" * 50)

    def _setup_wandb(self):
        """Initialize Weights & Biases tracking if enabled in the configuration."""
        if self.config["training"]["use_wandb"]:
            import wandb

            wandb.init(
                project=self.config["training"].get(
                    "wandb_project", "CIFNet-Experiment"
                ),
                config=self.config,
                dir=str(self.output_dir),
                name=self.config["training"].get("wandb_run_name", None),
                reinit=True,
            )

            if self.config["training"].get("wandb_watch_model", False):
                wandb.watch(self.model, log="all", log_freq=100)

            logger.info("Weights & Biases tracking initialized.")
        else:
            logger.info("Weights & Biases tracking is disabled.")

    def run(self) -> Dict[str, float]:
        """Execute full experiment pipeline"""
        try:
            logger.info("Starting experiment")
            self._log_config()

            # Initialize components
            train_dataset, test_dataset = get_dataset_instance(
                self.config["dataset"]["name"]
            )
            self.model = build_incremental_model(self.config)
            self.trainer = CILTrainer(model=self.model, config=self.config)
            num_tasks = self.config["incremental"]["num_tasks"]

            # Initialize tracking
            self.task_accuracies = {i: [] for i in range(num_tasks)}

            self._setup_wandb()

            # Track emissions
            if self.config["training"].get("use_codecarbon", False):
                with EmissionsTracker(
                    project_name=f"{self.config['dataset']['name']}_inc",
                    output_dir=str(self.emissions_dir),
                ) as tracker:
                    self._run_training_loop(
                        num_tasks, train_dataset, test_dataset
                    )
            else:
                self._run_training_loop(
                    num_tasks, train_dataset, test_dataset
                )

            cl_metrics = calculate_cl_metrics(self.task_accuracies)

            if self.config["training"].get("use_wandb", False):
                import wandb
                # Flatten the cl_metrics dictionary if needed
                wandb_log = {
                    "final_accuracy": cl_metrics["final_accuracy"],
                    "mean_accuracy": np.mean(cl_metrics["mean_accuracy"]),
                }

                for i, acc in enumerate(cl_metrics["mean_accuracy"]):
                    wandb.log({"accuracy": acc}, step = i)

                if self.config["training"].get("use_codecarbon", False):
                    emissions_file = self.emissions_dir / "emissions.csv"
                    if emissions_file.exists():
                        emissions_df = pd.read_csv(emissions_file)
                        latest_run = emissions_df.iloc[-1]
                        duration_min = latest_run["duration"] / 60
                        emissions_kg = latest_run["emissions"]
                        energy_kwh = latest_run["energy_consumed"]

                        wandb_log.update({
                            "training_time_min": duration_min,
                            "emissions_kg": emissions_kg,
                            "energy_consumed_kwh": energy_kwh,
                        })

                wandb.log(wandb_log)

                wandb.finish()

            self._save_results(cl_metrics, self.task_accuracies)
            self._generate_plots(cl_metrics, self.task_accuracies)

            return cl_metrics

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise

    def _run_training_loop(
        self,
        num_tasks: int,
        train_dataset: Subset,
        test_dataset: Subset,
    ):
        """Core training loop implementation"""
        for task in range(num_tasks):
            test_metrics = self.trainer.train_task(task, train_dataset, test_dataset)

            # Update metrics
            for eval_task in range(task + 1):
                accuracy = test_metrics["accuracy"][eval_task]
                self.task_accuracies[eval_task].append(accuracy)

    def _save_results(self, cl_metrics: Dict, task_accuracies: Dict[int, List[float]]):
        """Save results with additional metrics"""
        result_data = {
            "final_accuracy": cl_metrics["final_accuracy"],
            "mean_accuracies": cl_metrics["mean_accuracy"],
            "final_accuracies": cl_metrics["final_accuracies"],
            "task_accuracies": task_accuracies,
            **self._get_hyperparams(),
            "emissions": self._get_emissions_data(),
        }

        # Save JSON files
        (self.output_dir / "results.json").write_text(json.dumps(result_data, indent=4))
        (self.output_dir / "detailed_metrics.json").write_text(
            json.dumps(task_accuracies, indent=4)
        )

    def _get_hyperparams(self) -> Dict:
        """Extract relevant hyperparameters for saving"""
        return {
            "dataset": self.config["dataset"],
            "model": self.config["model"],
            "training": self.config["training"],
            "incremental": self.config["incremental"],
        }

    def _get_emissions_data(self) -> Dict:
        """Read emissions data from CodeCarbon output"""
        emissions_file = next(self.emissions_dir.glob("*.csv"), None)
        if emissions_file:
            return {"emissions_file": str(emissions_file)}
        return {}

    def _generate_plots(
        self, cl_metrics: Dict, task_accuracies: Dict[int, List[float]]
    ):
        """Generate and save all required plots"""
        # Task progression plot
        task_plot_path = self.output_dir / "task_accuracy_progression.png"
        plot_task_progression(task_accuracies, str(task_plot_path))

        # Mean accuracy plot
        mean_plot_path = self.output_dir / "mean_accuracy_progression.png"
        plot_mean_accuracy_progression(cl_metrics["mean_accuracy"], str(mean_plot_path))

        logger.info(f"Saved task progression plot to: {task_plot_path}")
        logger.info(f"Saved mean accuracy plot to: {mean_plot_path}")


if __name__ == "__main__":
    # try:
    runner = ExperimentRunner()
    metrics = runner.run()
    logger.info("\nExperiment completed successfully!")
    logger.info(f"Final metrics: {json.dumps(metrics, indent=4)}")

    # except Exception as e:
    # logger.critical(f"Critical error: {str(e)}")
    # exit(1)
