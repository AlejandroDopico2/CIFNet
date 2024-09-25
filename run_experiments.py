import os
import itertools
import subprocess
from datetime import datetime
import json


def run_experiments():
    # Define the parameter combinations you want to experiment with
    # datasets = ["MNIST", "CIFAR10", "CIFAR100"]
    datasets = ["CIFAR100"]
    backbones = ["ResNet", "MobileNet", "Custom"]
    batch_sizes = [200, 500]
    rolann_lambs = [0.01, 0.1, 1.0]
    learning_rates = [0.001, 0.01, 0.1]
    dropout_rates = [0, 0.3]

    # Generate all combinations
    combinations = list(
        itertools.product(
            datasets,
            backbones,
            batch_sizes,
            rolann_lambs,
            learning_rates,
            dropout_rates,
        )
    )

    # Create a base directory for all experiments
    base_dir = f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(base_dir, exist_ok=True)

    # Dictionary to store all results
    all_results = {}

    # Run experiments for each combination
    for combo in combinations:
        dataset, backbone, batch_size, rolann_lamb, lr, dropout = combo

        # Create a directory for this specific experiment
        exp_name = (
            f"{dataset}_{backbone}_b{batch_size}_l{rolann_lamb}_lr{lr}_d{dropout}"
        )
        exp_dir = os.path.join(base_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        # Construct the command
        cmd = [
            "python",
            "main.py",
            "--dataset",
            dataset,
            "--backbone",
            backbone,
            "--batch_size",
            str(batch_size),
            "--epochs",
            str(30),
            "--rolann_lamb",
            str(rolann_lamb),
            "--learning_rate",
            str(lr),
            "--dropout_rate",
            str(dropout),
            "--pretrained",
            "--sparse",
            "--output_dir",
            exp_dir,
            "--num_instances",
            str(5000),
        ]

        # Run the experiment
        print(f"Running experiment: {exp_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Save the output
        with open(os.path.join(exp_dir, "output.log"), "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        # Extract the test accuracy from the output
        for line in reversed(result.stdout.split("\n")):
            if "Test Accuracy:" in line:
                test_accuracy = float(line.split(":")[1].strip())
                break
        else:
            test_accuracy = None

        for line in reversed(result.stdout.split("\n")):
            if "Train Accuracy:" in line:
                train_accuracy = float(line.split(":")[1].strip())
                break
        else:
            train_accuracy = None

        # Store the results
        if dataset not in all_results:
            all_results[dataset] = []
        all_results[dataset].append(
            {
                "backbone": backbone,
                "batch_size": batch_size,
                "rolann_lamb": rolann_lamb,
                "learning_rate": lr,
                "dropout_rate": dropout,
                "test_accuracy": test_accuracy,
                "train_accuracy": train_accuracy,
            }
        )

        print(f"Experiment completed: {exp_name}")
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Test Accuracy: {test_accuracy}")
        print("-" * 50)

    # Save all results to a JSON file
    with open(os.path.join(base_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Find and print the best hyperparameters for each dataset
    best_params = find_best_hyperparameters(all_results)
    print("\nBest Hyperparameters for Each Dataset:")
    print(json.dumps(best_params, indent=2))

    # Save best hyperparameters to a JSON file
    with open(os.path.join(base_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=2)


def find_best_hyperparameters(all_results):
    best_params = {}
    for dataset, results in all_results.items():
        best_result = max(results, key=lambda x: x["test_accuracy"])
        best_params[dataset] = {
            "backbone": best_result["backbone"],
            "batch_size": best_result["batch_size"],
            "rolann_lamb": best_result["rolann_lamb"],
            "learning_rate": best_result["learning_rate"],
            "dropout_rate": best_result["dropout_rate"],
            "test_accuracy": best_result["test_accuracy"],
        }
    return best_params


if __name__ == "__main__":
    run_experiments()
