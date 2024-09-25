import argparse
import os
from train import train
from data_utils import set_dataloaders
from model_utils import build_model
from plotting import plot_results
from config import get_config


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
        f"Backbone: {args.backbone} {'not' if not args.pretrained else ''} pretrained"
    )
    print(f"Binary Classification: {args.binary}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(
        f"ROLANN Lambda: {args.rolann_lamb} with dropout {args.dropout_rate} {'with reset after epoch' if args.reset else ''} {'and sparse mode' if args.sparse else ''}"
    )
    print(f"Use WandB: {'Enabled' if args.use_wandb else 'Disabled'}")

    train_loader, test_loader = set_dataloaders(config)
    model = build_model(config)

    results = train(model, train_loader, test_loader, config)
    print(f"Train Accuracy: {results['train_accuracy'][-1]:.4f}")
    print(f"Test Accuracy: {results['test_accuracy'][-1]:.4f}")

    plot_filename = f"{args.dataset}_{args.backbone}_plot.png"
    plot_path = os.path.join(args.output_dir, plot_filename)

    plot_results(results, save_path=plot_path)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
