from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results, save_path: Optional[str] = None):
    num_epochs = len(results["train_loss"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.plot(
        range(num_epochs), results["train_loss"], label="Train Loss", color="tab:red"
    )
    ax1.plot(
        range(num_epochs), results["test_loss"], label="Test Loss", color="tab:orange"
    )
    ax1.legend(loc="upper right")
    ax1.set_title("Training and Test Loss")

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.plot(
        range(num_epochs),
        results["train_accuracy"],
        label="Train Accuracy",
        color="tab:blue",
    )
    ax2.plot(
        range(num_epochs),
        results["test_accuracy"],
        label="Test Accuracy",
        color="tab:green",
    )
    ax2.legend(loc="upper right")
    ax2.set_title("Training and Test Accuracy")
    ax2.set_ylim(0, 1)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_overall_accuracy(overall_accuracies, num_classes_per_task, num_classes_total):
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(num_classes_per_task, num_classes_total + 1, num_classes_per_task),
        overall_accuracies,
        marker="o",
    )
    plt.title("Overall Accuracy After Each Task")
    plt.xlabel("Number of Classes Learned")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig("overall_accuracy.png")
    plt.close()
    print("Overall accuracy plot saved as 'overall_accuracy.png'")


def plot_task_accuracies(
    task_train_accuracies,
    task_accuracies,
    num_tasks,
    save_path: str = "task_accuracies.png",
):
    """
    Plots the accuracies for each task throughout training, ensuring all tasks end at the same
    number of learned classes and each starts at its respective point.

    Args:
        task_accuracies (dict): Dictionary where the key is the task index and the value is a list of accuracies.
        num_tasks (int): Total number of tasks to be plotted on the X-axis.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 5))

    for task, accuracies in task_accuracies.items():
        task += 1
        x = list(range(task, task + len(accuracies)))
        plt.plot(x, accuracies, marker="o", label=f"Task {task}")

    if task_train_accuracies is not None:
        for task, train_accuracy in task_train_accuracies.items():
            task += 1
            plt.plot(
                [task],
                [train_accuracy],
                marker="*",
                markersize=10,
                color="red",
                label=f"Task {task} (Train)",
            )

    plt.title("Task Accuracy Throughout Training")
    plt.xlabel("Number of Tasks Learned")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xlim(0.5, num_tasks + 0.5)
    plt.legend(loc=0)
    plt.savefig(save_path)
    plt.close()
    print(f"Task accuracies plot saved as {save_path}")


def plot_average_accuracy(
    task_accuracies,
    num_classes_per_task,
    save_path: str = "average_accuracy.png",
):
    mean_accuracies = []

    num_tasks = len(task_accuracies)

    cumulative_classes = [num_classes_per_task * (i + 1) for i in range(num_tasks)]

    for i in range(len(task_accuracies)):
        accuracies = [task_accuracies[str(j)][i - j] for j in range(i + 1)]
        mean_accuracies.append(sum(accuracies) / (i + 1))

    # Plotting with improved style and axis labels
    plt.figure(figsize=(10, 6))
    plt.plot(
        cumulative_classes,
        mean_accuracies,
        marker="o",
        color="b",
        label="Mean Accuracy",
    )

    # Enhanced labels and ticks
    plt.xlabel("Cumulative Number of Classes Learned", fontsize=12)
    plt.ylabel("Average Accuracy", fontsize=12)
    plt.title(
        "Average Accuracy vs. Cumulative Number of Classes Learned",
        fontsize=14,
        fontweight="bold",
    )

    # Setting axis limits and formatting
    plt.ylim(0, 1)  # accuracy values range from 0 to 1
    plt.xticks(cumulative_classes)  # cumulative classes on x-axis
    plt.yticks(np.linspace(0, 1, 11))  # create 0.1 intervals for better granularity

    # Add grid and legend
    plt.grid(visible=True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=10)

    # Save and show plot
    plt.savefig(save_path)
    plt.show()
