import argparse
import json
import os
from pathlib import Path

from loguru import logger
import numpy as np
from scripts.train import train
from utils.data_utils import get_datasets, get_transforms
from utils.model_utils import build_model
from utils.plotting import plot_task_accuracies
from config import get_batch_config

# Set up loguru logger
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CV model with a specified backbone and parameters."
    )

    # Dataset related arguments
    dataset_group = parser.add_argument_group(
        "Dataset", "Arguments related to dataset selection and configuration"
    )
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "CIFAR10", "CIFAR100"],
        help="Dataset to be used for training.",
    )
    dataset_group.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and testing datasets.",
    )
    dataset_group.add_argument(
        "--num_instances",
        type=int,
        default=None,
        help="Number of instances to use from the dataset (default: 5000)",
    )

    # Model and backbone related arguments
    model_group = parser.add_argument_group(
        "Model", "Arguments related to model architecture and configuration"
    )
    model_group.add_argument(
        "--backbone",
        type=str,
        choices=["ResNet", "MobileNet", "DenseNet", "Custom", "SmallResNet"],
        required=False,
        help="Backbone model type (e.g., ResNet, MobileNet).",
    )
    model_group.add_argument(
        "--pretrained",
        default=False,
        action="store_true",
        help="Use a pretrained backbone model.",
    )
    model_group.add_argument(
        "--binary",
        default=False,
        action="store_true",
        help="Use binary classification. If not set, the model will perform multi-class classification.",
    )
    model_group.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate term for the backbone model.",
    )
    model_group.add_argument(
        "--freeze_mode",
        type=str,
        default="all",
        choices=["none", "all", "partial"],
        help="Freezing mode for the backbone: none, all, or partial",
    )

    # ROLANN specific arguments
    rolann_group = parser.add_argument_group(
        "ROLANN", "Arguments specific to the ROLANN layer"
    )
    rolann_group.add_argument(
        "--rolann_lamb",
        type=float,
        default=0.01,
        help="Regularization term (lambda) for the ROLANN layer.",
    )
    rolann_group.add_argument(
        "--reset",
        default=False,
        action="store_true",
        help="Reset the ROLANN layer after each epoch.",
    )
    rolann_group.add_argument("--sparse", default=False, action="store_true")
    rolann_group.add_argument(
        "--dropout_rate",
        default=0.25,
        type=float,
        help="Dropout rate for the ROLANN layer.",
    )

    # Training process arguments
    training_group = parser.add_argument_group(
        "Training", "Arguments related to the training process"
    )
    training_group.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for training the model.",
    )
    training_group.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="Enable Weights & Biases (WandB) for experiment tracking.",
    )
    training_group.add_argument(
        "--num_tasks",
        type=int,
        default=5,
        help="Number of tasks in the incremental learning setup.",
    )
    training_group.add_argument(
        "--classes_per_task",
        type=int,
        default=2,
        help="Number of classes introduced in each task.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save output files and plots.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_batch_config(args)

    # Print parsed arguments
    logger.info(f"Dataset: {args.dataset}")
    logger.info(
        f"Backbone: {args.backbone} ({'pretrained' if args.pretrained else 'not pretrained'}) (Freeze mode: {args.freeze_mode})"
    )
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(
        f"ROLANN Lambda: {args.rolann_lamb} with dropout {args.dropout_rate} {'with reset after epoch' if args.reset else ''} {'and sparse mode' if args.sparse else ''}"
    )
    logger.info(f"Use WandB: {'Enabled' if args.use_wandb else 'Disabled'}")

    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Training on device {config['device']}")

    transforms = get_transforms(config["dataset"], config["flatten"])

    train_dataset, test_dataset, num_classes = get_datasets(
        config["dataset"],
        transform=transforms,
    )

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    config["num_classes"] = num_classes

    model = build_model(config)

    _, task_accuracies = train(model, train_dataset, test_dataset, config)

    mean_test_accuracy = np.mean([acc[0] for acc in task_accuracies.values()])
    logger.info(f"Test Accuracy: {mean_test_accuracy}")

    plot_filename = f"{args.dataset}_{args.backbone}_plot.png"
    plot_path = os.path.join(args.output_dir, plot_filename)

    plot_task_accuracies(
        None, task_accuracies, config["num_tasks"], save_path=plot_path
    )
    logger.info(f"Plot saved to: {plot_path}")

    arguments_path = os.path.join(args.output_dir, "config.json")
    results_path = os.path.join(args.output_dir, "results.json")

    with open(arguments_path, "w") as f:
        json.dump(config, f, indent=4)

    with open(results_path, "w") as f:
        json.dump(task_accuracies, f, indent=4)


if __name__ == "__main__":
    main()
