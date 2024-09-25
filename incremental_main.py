import argparse
import os
from data_utils import get_transforms
from experience_replay_incremental_train import trainER
from incremental_train import train
from incremental_data_utils import get_datasets
from model_utils import build_incremental_model
from plotting import plot_overall_accuracy, plot_results, plot_task_accuracies
from config import get_config
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a incremental CV model with a specified backbone and parameters."
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
    dataset_group.add_argument(
        "--num_instances",
        type=int,
        default=5000,
        help="Number of instances to use from the dataset (default: 5000)",
    )

    # Model and backbone related arguments
    model_group = parser.add_argument_group(
        "Model", "Arguments related to model architecture and configuration"
    )
    model_group.add_argument(
        "--backbone",
        type=str,
        choices=[
            "ResNet",
            "MobileNet",
            "DenseNet",
            "Custom",
        ],
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
        "--freeze",
        default=False,
        action="store_true",
        help="Freeze the backbone model during training.",
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
        default=100,
        help="Number samples per task.",
    )
    incremental_group.add_argument(
        "--buffer_size",
        type=int,
        default=100,
        help="Number of instances of each task to save in replay buffer.",
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


def main() -> None:
    args = parse_args()
    config = get_config(args)

    # Print parsed arguments
    print(f"Dataset: {args.dataset}")
    print(
        f"Backbone: {args.backbone} ({'pretrained' if args.pretrained else 'not pretrained'}) {'(Frozen)' if args.freeze else ''}"
    )
    print(f"Binary Classification: {'Enabled' if args.binary else 'Disabled'}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Instances: {args.num_instances}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")

    print(
        f"ROLANN Lambda: {args.rolann_lamb} | Dropout Rate: {args.dropout_rate} "
        f"{'(Reset after each epoch)' if args.reset else ''} "
        f"{'(Sparse Mode Enabled)' if args.sparse else ''}"
    )

    print(f"Number of Tasks: {args.num_tasks}")
    print(f"Classes per Task: {args.classes_per_task}")
    print(f"Initial Tasks: {args.initial_tasks}")
    print(f"Samples per Task: {config['samples_per_task']}")

    print(f"Weights & Biases: {'Enabled' if args.use_wandb else 'Disabled'}")
    print(f"Output Directory: {args.output_dir}")

    train_dataset, test_dataset = get_datasets(
        config["dataset"],
        binary=config["binary"],
        transform=get_transforms(config["dataset"], config["flatten"]),
    )
    model = build_incremental_model(config)

    filename = f"{args.dataset}_{args.backbone}"
    plot_path = os.path.join(args.output_dir, filename + "_plot.png")
    log_path = os.path.join(args.output_dir, "results.json")

    use_experience_replay = True

    if use_experience_replay:
        results, task_accuracies = trainER(model, train_dataset, test_dataset, config)
        desired_hyperparams = [
            "dataset",
            "backbone",
            "batch_size",
            "epochs",
            "rolann_lamb",
            "dropout_rate",
            "pretrained",
            "output_dir",
            "samples_per_task",
            "freeze",
            "num_tasks",
            "classes_per_task",
            "buffer_size"
        ]

        hyperparameters_to_save = {key: config[key] for key in desired_hyperparams if key in config}

        output_data = {
            "task_accuracies": task_accuracies,
            "hyperparameters": hyperparameters_to_save
        }

        with open(log_path, "w") as f:
            json.dump(output_data, f, indent=4)
    else:
        results = train(model, train_dataset, test_dataset, config)
    print(f"Train Accuracy: {results['train_accuracy'][-1]:.4f}")
    print(f"Test Accuracy: {results['test_accuracy'][-1]:.4f}")

    # Plotting
    # plot_overall_accuracy(results, config["num_classes_per_task"], config["num_classes_per_task"] * config["num_tasks"])
    plot_task_accuracies(task_accuracies, config["num_tasks"])

    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
