import argparse
import os
from pathlib import Path
from typing import Dict, Union
import pandas as pd
import json
from loguru import logger

from utils.incremental_data_utils import get_transforms
from scripts.experience_replay_incremental_train import train_ExpansionBuffer
from incremental_train import incremental_train
from utils.incremental_data_utils import get_datasets
from utils.model_utils import build_incremental_model
from utils.plotting import plot_task_accuracies
from config import get_continual_config
from utils.utils import calculate_cl_metrics

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
        default=256,
        help="Batch size for training and testing datasets.",
    )

    # Model and backbone related arguments
    model_group = parser.add_argument_group(
        "Model", "Arguments related to model architecture and configuration"
    )
    model_group.add_argument(
        "--backbone",
        type=str,
        choices=["ResNet", "SmallResNet", "MobileNet", "DenseNet", "Custom"],
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
        default=0.0,
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
        default=None,
        help="Number of instances of each task to save in replay buffer.",
    )
    incremental_group.add_argument(
        "--use_eb",
        default=False,
        action="store_true",
        help="Enable using the Expansion Buffer for avoiding interference.",
    )
    incremental_group.add_argument(
        "--sampling_strategy",
        type=str,
        choices=["centroid", "entropy", "kmeans", "random", "typicality", "boundary"],
        required=False,
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
    logger.info(f"Experiment Name: {config['name']}")
    logger.info(f"Device: {config['device']}")
    logger.info("="*50)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning Rate: {args.learning_rate:.6f}")
    logger.info("="*50)
    logger.info(f"Backbone: {args.backbone or 'None'} "
                f"({'Pretrained' if args.pretrained else 'Not Pretrained'}) | "
                f"Freeze Mode: {args.freeze_mode}")
    logger.info("="*50)
    logger.info(f"ROLANN Lambda: {args.rolann_lamb:.4f} "
                f"{'(Frozen)' if args.freeze_rolann else ''} | "
                f"Dropout Rate: {args.dropout_rate:.2f} "
                f"{'(Sparse Mode Enabled)' if args.sparse else ''}")
    logger.info("="*50)
    logger.info(f"Incremental Learning Setup:")
    logger.info(f" - Number of Tasks: {args.num_tasks}")
    logger.info(f" - Classes per Task: {args.classes_per_task}")
    logger.info(f" - Initial Tasks: {args.initial_tasks}")
    logger.info(f" - Samples per Task: {config['samples_per_task'] or 'All dataset'}")
    logger.info(f" - Buffer Size: {args.buffer_size or 'None'}")
    logger.info(f" - Expansion Buffer: {'Enabled' if args.use_eb else 'Disabled'}")
    logger.info("="*50)
    logger.info(f"Sampling Strategy: {args.sampling_strategy or 'Default'}")
    logger.info(f"Tracking with WandB: {'Enabled' if args.use_wandb else 'Disabled'}")
    logger.info(f"Output Directory: {args.output_dir}")


    transforms = get_transforms(config["dataset"], config["flatten"])

    train_dataset, test_dataset = get_datasets(
        config["dataset"],
        transform=transforms,
    )
    model = build_incremental_model(config)

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    filename = f"{args.dataset}_{args.backbone}"
    plot_path = os.path.join(args.output_dir, filename + "_plot.png")

    if args.use_eb:

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

    csv_path = os.path.join(".", "results_log.csv")
    df = pd.DataFrame.from_dict([log_data])

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    dir_path = os.path.join(args.output_dir, f"{args.dataset}_{args.backbone}")

    with open(dir_path + "_results.json", "w") as f:
        json.dump(log_data, f, indent=4)

    with open(dir_path + "_task_accuracies.json", "w") as f:
        json.dump(task_accuracies, f, indent=4)

    logger.info(f"Average Forgetting: {cl_metrics['avg_forgetting']:.4f}")
    logger.info(f"Average Retained Accuracy: {cl_metrics['avg_retained']:.4f}")
    logger.info(
        f"Average Final Accuracy: {cl_metrics['avg_final_accuracy']* 100 :.4f}% "
    )
    logger.info(f"Results appended to: {csv_path}")
    logger.info(f"Detailed results saved to: {dir_path}")

    # Plotting task accuracies
    plot_task_accuracies(
        task_train_accuracies, task_accuracies, config["num_tasks"], save_path=plot_path
    )
    logger.info(f"Plot saved to: {plot_path}")

    return log_data


if __name__ == "__main__":
    main()
