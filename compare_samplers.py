import os
import json
import re
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from loguru import logger

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

def get_buffer_sizes(directory):
    """Función que extrae los tamaños de buffer desde los nombres de los subdirectorios."""
    buffer_sizes = []
    print(directory)
    for subdir in os.listdir(directory):
        match = re.search(r'buffer_size_(\d+)', subdir)
        if match:
            buffer_size = int(match.group(1))
            buffer_sizes.append(buffer_size)
    return sorted(buffer_sizes)

def load_results(directory, buffer_sizes, metrics):
    results = {metric: [] for metric in metrics}

    for buffer_size in buffer_sizes:
        buffer_dir = os.path.join(directory, f"buffer_size_{buffer_size}_run_0")
        result_file = os.path.join(buffer_dir, f"CIFAR10_resNet_results.json")  # Cambiar según tu esquema de nombres

        if not os.path.exists(result_file):
            logger.error(f"Results file not found: {result_file}")
            continue

        with open(result_file, "r") as file:
            experiment_results = json.load(file)

        for metric in metrics:
            if metric in experiment_results:
                results[metric].append(experiment_results[metric])
            else:
                logger.warning(f"Metric {metric} not found in {result_file}")
                results[metric].append(None)

    return results

def plot_comparative_results(directories, metrics, output_dir):
    fig, axs = plt.subplots(2, 1, figsize=(16, 9))
    axs = axs.ravel()

    print(mcolors.TABLEAU_COLORS)
    colors = list(mcolors.TABLEAU_COLORS.keys())[:len(directories)]
    method_names = [os.path.basename(directory) for directory in directories]

    for i, metric in enumerate(metrics):
        for j, directory in enumerate(directories):
            buffer_sizes = get_buffer_sizes(directory)
            method_results = load_results(directory, buffer_sizes, metrics)

            # Filtrar valores nulos
            valid_means = [
                result for result in method_results[metric] if result is not None
            ]

            if not valid_means:
                continue

            # Graficar los resultados de cada método con un color diferente
            axs[i].plot(
                buffer_sizes[: len(valid_means)], 
                valid_means, 
                label=method_names[j], 
                color=colors[j % len(colors)], 
                marker='o'
            )
        
        axs[i].set_xlabel("Buffer Size")
        axs[i].set_ylabel(metric)
        axs[i].set_title(f"{metric} vs Buffer Size")
        axs[i].legend()

    plt.tight_layout()
    result_path = os.path.join(output_dir, "comparison_results.png")
    plt.savefig(result_path)
    logger.info(f"Comparative results plotted and saved to {result_path}")

if __name__ == "__main__":

    experiments_dir = os.path.join("experiments", "experiments_10_22")
    directories = os.listdir(experiments_dir)

    directories = [os.path.join(experiments_dir, method_dir) for method_dir in directories]

    metrics = [
        "avg_forgetting",
        "avg_final_accuracy",
    ]

    output_dir = os.path.join(experiments_dir, "comparative_experiments")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_comparative_results(directories, metrics, output_dir)
