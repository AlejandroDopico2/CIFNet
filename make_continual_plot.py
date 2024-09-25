from plotting import plot_task_accuracies
import json

folder = "experiments_20240920_132506\CIFAR10_ResNet_l0.1_d0.3"

with open(f"{folder}/results.json", "r") as f:
    task_accuracies = json.load(f)["task_accuracies"]

task_accuracies = {int(key): value for key, value in task_accuracies.items()}


plot_task_accuracies(task_accuracies, len(task_accuracies), "task_accuracies_cifar.png")
