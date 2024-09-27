import argparse
import os
from time import time
import numpy as np
from matplotlib import pyplot as plt
from loguru import logger
from incremental_main import main

logger.remove()  # Remove default logger to customize it
logger.add(
    lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def run_multiple_mains(buffer_sizes, num_runs=2):
    args = argparse.Namespace(
        dataset="MNIST",
        backbone="ResNet",
        pretrained=False,
        freeze_mode="all",
        binary=False,
        batch_size=32,
        epochs=1,
        learning_rate=0.001,
        rolann_lamb=0.01,
        dropout_rate=0.25,
        reset=False,
        sparse=False,
        num_tasks=5,
        classes_per_task=2,
        initial_tasks=1,
        use_er=True,
        use_wandb=False,
        samples_per_task=10000,
        each_step=False,
    )

    results = []
    for buffer_size in buffer_sizes:
        for each_step in [True, False]:  # Loop for each_step values
            buffer_results = []
            logger.info(
                f"Starting tests with buffer_size {buffer_size} and each_step {each_step}"
            )

            for run in range(num_runs):
                args.buffer_size = buffer_size
                args.each_step = each_step
                args.output_dir = os.path.join(
                    "buffer_evaluation",
                    f"buffer_size_{buffer_size}_each_step_{each_step}_run_{run}",
                )

                logger.info(
                    f"Running with buffer_size {buffer_size}, each_step {each_step}, run {run + 1}/{num_runs}"
                )
                start_time = time()

                try:
                    current_results = main(args)
                except Exception as e:
                    logger.error(f"Error during execution: {e}")
                    continue

                end_time = time()
                current_results["total_time"] = end_time - start_time
                logger.info(
                    f"Completed run {run + 1} in {current_results['total_time']:.2f} seconds"
                )
                buffer_results.append(current_results)

            mean_results = {}
            std_results = {}

            for key in buffer_results[0].keys():
                if isinstance(buffer_results[0][key], (int, float)):
                    values = [r[key] for r in buffer_results]
                    mean_results[key] = np.mean(values)
                    std_results[key] = np.std(values)
                else:
                    mean_results[key] = buffer_results[0][key]

            mean_results["buffer_size"] = buffer_size
            mean_results["each_step"] = each_step
            std_results["buffer_size"] = buffer_size
            std_results["each_step"] = each_step
            results.append((mean_results, std_results))

            logger.debug(
                f"Results for buffer_size {buffer_size}, each_step {each_step}: {mean_results}"
            )

    return results


def plot_results(results):
    buffer_sizes = sorted(list(set([r[0]["buffer_size"] for r in results])))
    metrics = [
        "avg_forgetting",
        "avg_retained_accuracy",
        "avg_final_accuracy",
        "total_time",
    ]

    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    axs = axs.ravel()

    # Split results by each_step value
    each_step_true_results = [r for r in results if r[0]["each_step"]]
    each_step_false_results = [r for r in results if not r[0]["each_step"]]

    for i, metric in enumerate(metrics):
        # Gather data for each_step=True
        means_true = [r[0][metric] for r in each_step_true_results]
        stds_true = [r[1][metric] for r in each_step_true_results]

        # Gather data for each_step=False
        means_false = [r[0][metric] for r in each_step_false_results]
        stds_false = [r[1][metric] for r in each_step_false_results]

        # Plot each_step=True
        axs[i].errorbar(
            buffer_sizes,
            means_true,
            yerr=stds_true,
            fmt="-o",
            label="each_step=True",
            color="blue",
        )

        # Plot each_step=False
        axs[i].errorbar(
            buffer_sizes,
            means_false,
            yerr=stds_false,
            fmt="-o",
            label="each_step=False",
            color="orange",
        )

        axs[i].set_xlabel("Buffer Size")
        axs[i].set_ylabel(metric)
        axs[i].set_title(f"{metric} vs Buffer Size")
        axs[i].legend()  # Add legend to differentiate each_step values

    plt.tight_layout()
    plt.savefig(os.path.join("buffer_evaluation", "buffer_size_results_comparison.png"))
    logger.info("Results plotted and saved to buffer_size_results_comparison.png")


if __name__ == "__main__":
    buffer_sizes = range(100, 501, 100)
    results = run_multiple_mains(buffer_sizes, num_runs=5)
    plot_results(results)
