import json
import os
import re


def extract_learning_rate(plot_line):
    # Busca la parte "lr<valor>" en la cadena
    match = re.search(r"lr([0-9.]+)", plot_line)
    if match:
        return float(match.group(1))
    return None


def parse_log_file(log_content):
    result = {}

    result["dataset"] = re.search(r"Dataset: (\w+)", log_content).group(1)
    result["backbone"] = re.search(r"Backbone: (.+)", log_content).group(1).strip()
    result["batch_size"] = int(re.search(r"Batch Size: (\d+)", log_content).group(1))
    result["rolann_lambda"] = float(
        re.search(r"ROLANN Lambda: ([\d.]+)", log_content).group(1)
    )
    result["dropout"] = float(re.search(r"with dropout ([\d.]+)", log_content).group(1))

    # Extraer todas las precisiones de prueba
    test_accuracies = re.findall(r"Test Accuracy: ([\d.]+)", log_content)
    if test_accuracies:
        result["test_accuracy"] = float(
            test_accuracies[-1]
        )  # Tomar la última precisión de prueba

    # Extraer la precisión final de entrenamiento
    result["train_accuracy"] = float(
        re.search(r"Train Accuracy: ([\d.]+)", log_content).group(1)
    )

    plot_line = re.search(r"Plot saved to: (.+)", log_content)
    if plot_line:
        plot_path = plot_line.group(1)
        learning_rate = re.search(r"lr([\d.]+)", plot_path)
        if learning_rate:
            result["learning_rate"] = float(learning_rate.group(1))

    return result


def extract_experiments_from_logs(log_dir):
    all_experiments = []
    for experiment in os.listdir(log_dir):
        log_file = os.path.join(log_dir, experiment, "output.log")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log_content = f.read()
            experiment_data = parse_log_file(log_content)
            all_experiments.append(experiment_data)

    return all_experiments


def save_to_json(data, output_file):
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


def main():
    log_dir = "experiments_20240905_141154"  # Actualiza esta ruta si es necesario
    output_file = "experiment_results.json"

    experiments = extract_experiments_from_logs(log_dir)
    save_to_json(experiments, output_file)
    print(f"Results have been saved to '{output_file}'.")


if __name__ == "__main__":
    main()
