import argparse
import os
from pathlib import Path
from typing import Dict, Union
import json
import yaml
from loguru import logger

from utils.incremental_data_utils import get_transforms
from scripts.experience_replay_incremental_train import train_ExpansionBuffer
from incremental_train import incremental_train
from utils.incremental_data_utils import get_datasets
from utils.model_utils import build_incremental_model
from utils.plotting import plot_task_accuracies
from utils.utils import calculate_cl_metrics

# Set up loguru logger
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

def load_yaml_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an incremental CV model with a specified backbone and parameters."
    )
    parser.add_argument(
        "-p",
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    
    return parser.parse_args()

def main(config=None) -> Dict[str, Union[float, str]]:
    if config is None:
        config = load_yaml_config(config.config_path)


    # Logging parsed arguments
    logger.info(f"Device: {config['device']}")
    logger.info("="*50)
    logger.info(f"Dataset: {config['dataset']['name']}")
    logger.info(f"Batch Size: {config['dataset']['batch_size']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Learning Rate: {config['model']['learning_rate']:.6f}")
    logger.info("="*50)
    logger.info(f"Backbone: {config['model']['backbone'] or 'None'} "
                f"({'Pretrained' if config['model']['pretrained'] else 'Not Pretrained'}) | "
                f"Freeze Mode: {config['model']['freeze_mode']}")
    logger.info("="*50)
    logger.info(f"ROLANN Lambda: {config['rolann']['rolann_lamb']:.4f} "
                f"{'(Frozen)' if config['rolann']['freeze_rolann'] else ''} | "
                f"Dropout Rate: {config['rolann']['dropout_rate']:.2f} "
                f"{'(Sparse Mode Enabled)' if config['rolann']['sparse'] else ''}")
    logger.info("="*50)
    logger.info(f"Incremental Learning Setup:")
    logger.info(f" - Number of Tasks: {config['incremental']['num_tasks']}")
    logger.info(f" - Classes per Task: {config['incremental']['classes_per_task']}")
    logger.info(f" - Initial Tasks: {config['incremental']['initial_tasks']}")
    logger.info(f" - Samples per Task: {config['incremental']['samples_per_task'] or 'All dataset'}")
    logger.info(f" - Buffer Size: {config['incremental']['buffer_size'] or 'None'}")
    logger.info(f" - Expansion Buffer: {'Enabled' if config['incremental']['use_eb'] else 'Disabled'}")
    logger.info("="*50)
    logger.info(f"Sampling Strategy: {config['incremental']['sampling_strategy'] or 'Default'}")
    logger.info(f"Tracking with WandB: {'Enabled' if config['training']['use_wandb'] else 'Disabled'}")
    logger.info(f"Output Directory: {config['output_dir']}")

    flatten = True if config['model']['backbone'] is None else False

    transforms = get_transforms(config["dataset"]["name"], flatten)

    train_dataset, test_dataset = get_datasets(
        config["dataset"]["name"],
        transform=transforms,
    )
    model = build_incremental_model(config)

    if config["incremental"]["use_eb"]:

        results, task_train_accuracies, task_accuracies = train_ExpansionBuffer(
            model, train_dataset, test_dataset, config
        )
    else:
        results, task_train_accuracies, task_accuracies = incremental_train(
            model, train_dataset, test_dataset, config
        )

    cl_metrics = calculate_cl_metrics(task_accuracies)

    hyperparameters_to_save = {
        key: config[key]
        for key in config
        if key
        in [
            "dataset",
            "backbone",
            "batch_size",
            "epochs",
            "rolann_lamb",
            "dropout_rate",
            "pretrained",
            "output_dir",
            "samples_per_task",
            "freeze_mode",
            "num_tasks",
            "classes_per_task",
            "buffer_size",
            "use_er",
            "each_step",
        ]
    }

    log_data = {
        "avg_forgetting": cl_metrics["avg_forgetting"],
        "avg_retained_accuracy": cl_metrics["avg_retained"],
        "avg_final_accuracy": cl_metrics["avg_final_accuracy"],
        **hyperparameters_to_save,
    }

    Path(config['output_dir']).mkdir(exist_ok=True, parents=True)

    filename = f"{config['dataset']['name']}_{config['model']['backbone']}"
    plot_path = os.path.join(config['output_dir'], filename + "_plot.png")

    dir_path = os.path.join(config['output_dir'], filename)

    with open(dir_path + "_results.json", "w") as f:
        json.dump(log_data, f, indent=4)

    with open(dir_path + "_task_accuracies.json", "w") as f:
        json.dump(task_accuracies, f, indent=4)

    logger.info(f"Average Forgetting: {cl_metrics['avg_forgetting']:.4f}")
    logger.info(f"Average Retained Accuracy: {cl_metrics['avg_retained']:.4f}")
    logger.info(
        f"Average Final Accuracy: {cl_metrics['avg_final_accuracy']* 100 :.4f}% "
    )
    # logger.info(f"Results appended to: {csv_path}")
    logger.info(f"Detailed results saved to: {dir_path}")

    # Plotting task accuracies
    plot_task_accuracies(
        task_train_accuracies, task_accuracies, config["incremental"]["num_tasks"], save_path=plot_path
    )
    logger.info(f"Plot saved to: {plot_path}")

    return log_data


if __name__ == "__main__":
    main()
