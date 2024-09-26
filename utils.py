# ----------------------------------------------------------------
#   Funciones Auxiliares
# ----------------------------------------------------------------

import os
import struct
from typing import Dict, List

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from models.RolanNET import RolanNET


def check_equals(params1, params2):
    rm, ru, rs = params1
    fm, fu, fs = params2

    checks = {
        "m": torch.equal(fm, rm),
        "u": torch.equal(fu, ru),
        "s": torch.equal(fs, rs),
    }

    print("\nParameter Comparison:")
    for name, result in checks.items():
        status = "MATCH" if result else "MISMATCH"
        print(f" - {name}: {status}")

    if not all(checks.values()):
        print("Invalid parameters detected!")


def check_weights(rw, fw):
    print("\nWeights Comparison:")
    weights_equal = torch.allclose(fw, rw, atol=1e-03)
    status = "MATCH" if weights_equal else "MISMATCH"
    print(f" - Weights: {status}")


def check_outputs(ro, fo, debug=True):
    r_y = torch.argmax(ro, dim=1)
    f_y = torch.argmax(fo, dim=0)

    outputs_equal = torch.equal(r_y, f_y)
    if debug:
        print("\nOutputs Comparison:")
        status = "MATCH" if outputs_equal else "MISMATCH"
        print(f" - Outputs: {status}")

    return r_y, f_y


def compare_accuracies(test_labels, r_y, f_y):
    federated_accuracy = 100 * accuracy_score(
        test_labels.cpu().numpy(), f_y.cpu().numpy()
    )
    rolann_accuracy = 100 * accuracy_score(test_labels.cpu().numpy(), r_y.cpu().numpy())

    print("\nAccuracies:")
    print(f" - Federated Model Accuracy: {federated_accuracy:.2f}%")
    print(f" - Regular Model Accuracy:   {rolann_accuracy:.2f}%")

    return federated_accuracy, rolann_accuracy


def load_mnist(path, kind="train"):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte")

    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
        images = images.astype(np.float32) / 255.0

    return images, labels


def build_model(
    dataset: str,
    backbone: str,
    binary: bool = False,
    pretrained: bool = True,
    device: str = "cuda",
    rolann_lamb: float = 0.01,
    f_act: str = "logs",
):
    in_channels = 1 if dataset == "MNIST" else 3

    model = RolanNET(
        num_classes=2 if binary else 10,
        activation=f_act,
        lamb=rolann_lamb,
        pretrained=pretrained,
        backbone=backbone,
        in_channels=in_channels,
        sparse=False,
    ).to(device)

    return model


def process_labels(labels: torch.Tensor):
    return labels * 0.9 + 0.05


def get_prediction(client, coordinator, test_imgs, test_labels):
    client.set_params(coordinator.send_weights())
    test_y = client(test_imgs)

    acc = 100 * accuracy_score(
        test_labels.cpu().numpy(), torch.argmax(test_y, dim=0).cpu().numpy()
    )
    return acc


def plot_lambda_values(lambda_values, lambda_accuracies):
    plt.figure(figsize=(10, 6))

    plt.bar(
        lambda_values,
        list(lambda_accuracies.values()),
        width=0.8,
        color="#3498db",
        label="Federated Model",
    )

    plt.xlabel("Lambda Value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Lambda")
    plt.ylim(0, 100)
    plt.xscale("log")
    plt.legend(loc="upper left")

    plt.tight_layout()

    plt.show()


def plot_accuracies(lambda_values, lambda_accuracies):
    plt.figure(figsize=(10, 6))

    n_clients = len(lambda_accuracies[lambda_values[0]])

    for lamb in lambda_values:
        plt.plot(
            list(range(n_clients)),
            lambda_accuracies[lamb],
            marker="o",
            label=f"Lambda {lamb}",
        )

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Evolution Across Epochs for Different Lambda Values")
    plt.legend(title="Lambda Values")
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()


def calculate_cl_metrics(task_accuracies: Dict[int, List[float]]):
    num_tasks = len(task_accuracies.keys())

    forgetting_measures = []
    retained_accuracies = []

    for i in range(num_tasks - 1):
        max_accuracy = max(task_accuracies[i])
        final_accuracy = task_accuracies[i][-1]

        forgetting = max_accuracy - final_accuracy

        initial_accuracy = task_accuracies[i][0]

        if initial_accuracy > 1e-6:
            retained = final_accuracy / initial_accuracy
        else:
            retained = 0

        forgetting_measures.append(forgetting)
        retained_accuracies.append(retained)

    avg_forgetting = np.mean(forgetting_measures) if forgetting_measures else 0
    avg_retained = np.mean(retained_accuracies) if retained_accuracies else 0

    avg_final_accuracy = np.mean(
        [accuracy[-1] for accuracy in task_accuracies.values()]
    )

    return {
        "forgetting_measures": forgetting_measures,
        "retained_accuracies": retained_accuracies,
        "avg_forgetting": avg_forgetting,
        "avg_retained": avg_retained,
        "avg_final_accuracy": avg_final_accuracy,
    }
