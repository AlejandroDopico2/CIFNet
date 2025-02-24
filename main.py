import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from loguru import logger
from codecarbon import EmissionsTracker

from utils.data_utils import get_dataset_instance
from scripts.train_cifnet import CILTrainer
from utils.model_utils import build_incremental_model
from utils.plotting import plot_task_accuracies
from utils.utils import calculate_cl_metrics

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
            "-p", "--config_path", type=str, required=True,
            help="Path to YAML configuration file"
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
            'device', 'dataset', 'model', 'training',
            'incremental', 'output_dir', 'rolann'
        }
        if not required_keys.issubset(self.config.keys()):
            missing = required_keys - self.config.keys()
            raise ValueError(f"Missing config keys: {missing}")

    def _init_paths(self):
        """Initialize output directories"""
        self.output_dir = Path(self.config['output_dir'])
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
        """Initialize Weights & Biases tracking if enabled"""
        if self.config['training']['use_wandb']:
            import wandb
            wandb.init(
                project="RolanNet-Model",
                config=self.config,
                dir=str(self.output_dir)
            )

            wandb.watch(self.model)

    def run(self) -> Dict[str, float]:
        """Execute full experiment pipeline"""
        try:
            self._log_config()
            
            # Initialize components
            train_dataset, test_dataset = get_dataset_instance(
                self.config['dataset']['name']
            )
            self.model = build_incremental_model(self.config)
            self.trainer = CILTrainer(model=self.model, config=self.config)
            
            # Track emissions
            with EmissionsTracker(
                project_name=f"{self.config['dataset']['name']}_inc",
                output_dir=str(self.emissions_dir)
            ) as tracker:
                
                # Execute training
                results, train_acc, task_acc = self.trainer.train(
                    train_dataset, test_dataset
                )

            print(results, train_acc, task_acc)
            # Calculate and log metrics
            cl_metrics = calculate_cl_metrics(task_acc)
            self._save_results(cl_metrics, task_acc)
            self._generate_plots(train_acc, task_acc)

            return cl_metrics

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise

    def _save_results(self, cl_metrics: Dict, task_acc: Dict):
        """Save results to output directory"""
        result_data = {
            **cl_metrics,
            **self._get_hyperparams(),
            "emissions": self._get_emissions_data()
        }

        # Save JSON files
        (self.output_dir / "results.json").write_text(
            json.dumps(result_data, indent=4)
        )
        (self.output_dir / "task_accuracies.json").write_text(
            json.dumps(task_acc, indent=4)
        )

    def _get_hyperparams(self) -> Dict:
        """Extract relevant hyperparameters for saving"""
        return {
            'dataset': self.config['dataset'],
            'model': self.config['model'],
            'training': self.config['training'],
            'incremental': self.config['incremental']
        }

    def _get_emissions_data(self) -> Dict:
        """Read emissions data from CodeCarbon output"""
        emissions_file = next(self.emissions_dir.glob("*.csv"), None)
        if emissions_file:
            return {"emissions_file": str(emissions_file)}
        return {}

    def _generate_plots(self, train_acc: Dict, task_acc: Dict):
        """Generate and save accuracy plots"""
        plot_path = self.output_dir / "accuracy_plot.png"
        plot_task_accuracies(
            train_acc,
            task_acc,
            self.config['incremental']['num_tasks'],
            save_path=str(plot_path)
        )
        logger.info(f"Saved accuracy plot to: {plot_path}")

if __name__ == "__main__":
    try:
        runner = ExperimentRunner()
        metrics = runner.run()
        logger.info("\nExperiment completed successfully!")
        logger.info(f"Final metrics: {json.dumps(metrics, indent=4)}")
        
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        exit(1)