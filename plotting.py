from typing import Optional
import matplotlib.pyplot as plt

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
