from glob import glob
import json
import os
from time import strftime, time
import numpy as np
from matplotlib import pyplot as plt
from loguru import logger
import yaml
from incremental_main import main

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def load_config(yaml_path):
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)


def create_output_directory(yaml_path):
    filename = os.path.basename(yaml_path).split(".")[0]

    dataset, strategy = filename.split("_")

    date_str = strftime("%m_%d")
    output_dir = os.path.join(
        "experiments", f"experiments_{date_str}", f"{dataset}_{strategy}"
    )

    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def run_multiple_mains(config, buffer_sizes, output_dir, num_runs=2):
    results = []
    for buffer_size in buffer_sizes:
        buffer_results = []
        logger.info(f"Starting tests with buffer_size {buffer_size}")

        for run in range(num_runs):
            config["incremental"]["buffer_size"] = buffer_size
            config["output_dir"] = os.path.join(
                output_dir,
                f"buffer_size_{buffer_size}_run_{run}",
            )

            logger.info(
                f"Running with buffer_size {buffer_size}, run {run + 1}/{num_runs}"
            )
            start_time = time()

            try:
                current_results = main(config)
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


def plot_results(results, output_dir):
    print(results)
    buffer_sizes = sorted(set(r[0]["buffer_size"] for r in results))

    metrics = [
        "avg_forgetting",
        # "avg_retained_accuracy",
        "avg_final_accuracy",
        # "total_time",
    ]

    fig, axs = plt.subplots(2, 1, figsize=(16, 9))
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
            color="blue",
        )

        axs[i].set_xlabel("Buffer Size")
        axs[i].set_ylabel(metric)
        axs[i].set_title(f"{metric} vs Buffer Size")
        axs[i].legend()

    plt.tight_layout()
    result_path = os.path.join(output_dir, "buffer_size_results_comparison.png")
    plt.savefig(result_path)
    logger.info("Results plotted and saved to buffer_size_results_comparison.png")


def process_experiments(base_dir):
    results = []

    # Recorre todos los directorios en experiments_10_24/CIFAR10_random/
    for dirpath in glob(os.path.join(base_dir, "buffer_size_*_run_*")):
        # Lee el archivo CIFAR10_ResNet_results.json
        json_path = os.path.join(dirpath, "CIFAR10_ResNet_results.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                buffer_size = int(dirpath.split("_")[-3])  # Obtiene el tamaño de buffer
                avg_forgetting = data.get("avg_forgetting", 0)
                avg_final_accuracy = data.get("avg_final_accuracy", 0)

                results.append(
                    [
                        {
                            "buffer_size": buffer_size,
                            "avg_forgetting": avg_forgetting,
                            "avg_final_accuracy": avg_final_accuracy,
                        }
                    ]
                )

    return results


if __name__ == "__main__":
    yaml_path = "cfgs/CIFAR10_random.yaml"
    config = load_config(yaml_path)

    output_dir = create_output_directory(yaml_path)

    # buffer_sizes = range(900, 1501, 200)
    # results = run_multiple_mains(config, buffer_sizes, output_dir, num_runs=2)

    results = process_experiments(
        os.path.join("experiments", "experiments_10_24", "CIFAR10_random")
    )

    print(output_dir)
    plot_results(results, output_dir)
