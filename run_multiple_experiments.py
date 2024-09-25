import os
import subprocess
from datetime import datetime
import json
import multiprocessing
import optuna


def run_experiment(trial):
    dataset = "CIFAR100"

    # Define hyperparameters to search
    backbone = trial.suggest_categorical("backbone", ["ResNet", "MobileNet", "Custom"])
    batch_size = trial.suggest_int("batch_size", 200, 500, step=150)
    rolann_lamb = trial.suggest_float("rolann_lamb", 0.01, 1.0, log=True)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
    dropout = trial.suggest_float("dropout_rate", 0.0, 0.5)

    # Create a directory for this specific experiment
    exp_name = f"{dataset}_{backbone}_b{batch_size}_l{rolann_lamb}_lr{learning_rate}_d{dropout}"
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
        str(learning_rate),
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

    # Extract accuracies
    test_accuracy, train_accuracy = None, None
    for line in reversed(result.stdout.split("\n")):
        if "Test Accuracy:" in line:
            test_accuracy = float(line.split(":")[1].strip())
            break
    for line in reversed(result.stdout.split("\n")):
        if "Train Accuracy:" in line:
            train_accuracy = float(line.split(":")[1].strip())
            break

    # Return the negative test accuracy (minimization problem)
    return -test_accuracy


def objective(trial):
    return run_experiment(trial)


def run_bayesian_search(n_trials, n_jobs):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Save the best hyperparameters
    best_params = study.best_params
    print("\nBest Hyperparameters:")
    print(json.dumps(best_params, indent=2))

    with open(os.path.join(base_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=2)


if __name__ == "__main__":
    # Create a base directory for all experiments
    base_dir = f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(base_dir, exist_ok=True)

    # Run Bayesian optimization
    run_bayesian_search(n_trials=50, n_jobs=multiprocessing.cpu_count() // 2)
