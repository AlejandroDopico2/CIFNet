import argparse
import os
from time import time
import numpy as np
from matplotlib import pyplot as plt
from loguru import logger
from incremental_main_argparse import main

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def run_multiple_mains(buffer_sizes, num_runs=2):
    args = argparse.Namespace(
        dataset="MNIST",
        backbone="ResNet",
        pretrained=True,
        freeze_mode="all",
        batch_size=300,
        epochs=1,
        learning_rate=0.001,
        rolann_lamb=0.01,
        dropout_rate=0.0,
        sparse=False,
        num_tasks=5,
        classes_per_task=2,
        initial_tasks=1,
        use_eb=True,
        use_wandb=False,
        samples_per_task=None,
        freeze_rolann=False,
        sampling_strategy="centroid",
    )

    results = []
    for buffer_size in buffer_sizes:
        buffer_results = []
        logger.info(f"Starting tests with buffer_size {buffer_size}")

        for run in range(num_runs):
            args.buffer_size = buffer_size
            args.output_dir = os.path.join(
                "centroid_evaluation",
                f"buffer_size_{buffer_size}_run_{run}",
            )

            logger.info(
                f"Running with buffer_size {buffer_size}, run {run + 1}/{num_runs}"
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
        std_results["buffer_size"] = buffer_size
        results.append((mean_results, std_results))

        logger.debug(f"Results for buffer_size {buffer_size}: {mean_results}")

    return results


def plot_results(results):
    buffer_sizes = sorted(set(r[0]["buffer_size"] for r in results))

    metrics = [
        "avg_forgetting",
        "avg_retained_accuracy",
        "avg_final_accuracy",
        "total_time",
    ]

    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    axs = axs.ravel()

    for i, metric in enumerate(metrics):

        means = []
        stds = []

        for buffer_size in buffer_sizes:

            filtered_results = [
                r for r in results if r[0]["buffer_size"] == buffer_size
            ]

            if filtered_results:

                mean_value = sum(r[0][metric] for r in filtered_results) / len(
                    filtered_results
                )
                std_value = (
                    sum((r[0][metric] - mean_value) ** 2 for r in filtered_results)
                    / len(filtered_results)
                ) ** 0.5
            else:
                mean_value = 0
                std_value = 0

            means.append(mean_value)
            stds.append(std_value)

        axs[i].errorbar(
            buffer_sizes,
            means,
            yerr=stds,
            fmt="-o",
            label="each_step=True",
            color="blue",
        )

        axs[i].set_xlabel("Buffer Size")
        axs[i].set_ylabel(metric)
        axs[i].set_title(f"{metric} vs Buffer Size")
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "experiments",
            "experiments_10-07",
            "buffer_evaluation",
            "buffer_size_results_comparison.png",
        )
    )
    logger.info("Results plotted and saved to buffer_size_results_comparison.png")


if __name__ == "__main__":
    buffer_sizes = range(100, 1501, 200)
    results = run_multiple_mains(buffer_sizes, num_runs=1)
    plot_results(results)
