import argparse
import os
from pathlib import Path
from typing import Dict, Union
import pandas as pd
import json
from loguru import logger

from data_utils import get_transforms
from experience_replay_incremental_train import train_ER_AfterEpoch, train_ER_EachStep
from incremental_train import incremental_train
from incremental_data_utils import get_datasets
from model_utils import build_incremental_model
from plotting import plot_task_accuracies
from config import get_continual_config
from utils import calculate_cl_metrics

# Set up loguru logger
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an incremental CV model with a specified backbone and parameters."
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
        default=64,
        help="Batch size for training and testing datasets.",
    )

    # Model and backbone related arguments
    model_group = parser.add_argument_group(
        "Model", "Arguments related to model architecture and configuration"
    )
    model_group.add_argument(
        "--backbone",
        type=str,
        choices=["ResNet", "MobileNet", "DenseNet", "Custom"],
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
    rolann_group.add_argument("--sparse", default=False, action="store_true")
    rolann_group.add_argument(
        "--dropout_rate",
        default=0.25,
        type=float,
        help="Dropout rate for the ROLANN layer.",
    )
    rolann_group.add_argument(
        "--freeze_rolann",
        default=False,
        action="store_true",
        help="Freeze the ROLANN outputs layer during training.",
    )

    # Incremental learning specific arguments
    incremental_group = parser.add_argument_group(
        "Incremental Learning", "Arguments related to incremental learning"
    )
    incremental_group.add_argument(
        "--num_tasks",
        type=int,
        default=5,
        help="Number of tasks in the incremental learning setup.",
    )
    incremental_group.add_argument(
        "--classes_per_task",
        type=int,
        default=2,
        help="Number of classes introduced in each task.",
    )
    incremental_group.add_argument(
        "--initial_tasks",
        type=int,
        default=1,
        help="Number of tasks to start training.",
    )
    incremental_group.add_argument(
        "--samples_per_task",
        type=int,
        default=None,
        help="Number samples per task.",
    )
    incremental_group.add_argument(
        "--buffer_size",
        type=int,
        default=100,
        help="Number of instances of each task to save in replay buffer.",
    )
    incremental_group.add_argument(
        "--use_er",
        default=False,
        action="store_true",
        help="Enable using the Experience Replay Buffer for avoiding CF.",
    )
    incremental_group.add_argument(
        "--each_step",
        default=False,
        action="store_true",
        help="Enable using the Experience Replay each training step.",
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

    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save output files and plots.",
    )

    return parser.parse_args()


def main(args=None) -> Dict[str, Union[float, str]]:
    if args is None:
        args = parse_args()

    config = get_continual_config(args)

    # Logging parsed arguments
    logger.info(f"Dataset: {args.dataset}")
    logger.info(
        f"Backbone: {args.backbone} ({'pretrained' if args.pretrained else 'not pretrained'}) (Freeze mode: {args.freeze_mode})"
    )
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(
        f"ROLANN Lambda: {args.rolann_lamb} {'(freezed)' if args.freeze_rolann else ''} | Dropout Rate: {args.dropout_rate} "
        f"{'(Sparse Mode Enabled)' if args.sparse else ''}"
    )
    logger.info(f"Number of Tasks: {args.num_tasks}")
    logger.info(f"Classes per Task: {args.classes_per_task}")
    logger.info(f"Initial Tasks: {args.initial_tasks}")
    logger.info(f"Using Experience Replay: {args.use_er} {'each step' if args.each_step else 'each epoch.'}")
    logger.info(f"Samples per Task: {config['samples_per_task'] if config['samples_per_task'] else 'all the dataset.'}")
    logger.info(f"Weights & Biases: {'Enabled' if args.use_wandb else 'Disabled'}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Training on device {config['device']}")

    train_dataset, test_dataset = get_datasets(
        config["dataset"],
        transform=get_transforms(config["dataset"], config["flatten"]),
    )
    model = build_incremental_model(config)

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    filename = f"{args.dataset}_{args.backbone}"
    plot_path = os.path.join(args.output_dir, filename + "_plot.png")
    log_path = os.path.join(args.output_dir, "results.json")

    if args.use_er:

        if args.each_step:
            results, task_accuracies = train_ER_EachStep(
                model, train_dataset, test_dataset, config
            )
        else:
            results, task_accuracies = train_ER_AfterEpoch(
                model, train_dataset, test_dataset, config
            )
    else:
        results, task_accuracies = incremental_train(model, train_dataset, test_dataset, config)

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
            "each_step"
        ]
    }

    log_data = {
        "avg_forgetting": cl_metrics["avg_forgetting"],
        "avg_retained_accuracy": cl_metrics["avg_retained"],
        "avg_final_accuracy": cl_metrics["avg_final_accuracy"],
        **hyperparameters_to_save,
    }

    csv_path = os.path.join(".", "results_log.csv")
    df = pd.DataFrame.from_dict([log_data])

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    json_path = os.path.join(
        args.output_dir, f"{args.dataset}_{args.backbone}_results.json"
    )

    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=4)

    logger.info(f"Average Forgetting: {cl_metrics['avg_forgetting']:.4f}")
    logger.info(f"Average Retained Accuracy: {cl_metrics['avg_retained']:.4f}")
    logger.info(f"Results appended to: {csv_path}")
    logger.info(f"Detailed results saved to: {json_path}")

    # Plotting task accuracies
    plot_task_accuracies(task_accuracies, config["num_tasks"], save_path=plot_path)
    logger.info(f"Plot saved to: {plot_path}")

    return log_data


if __name__ == "__main__":
    main()
