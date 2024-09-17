"""
Example of using FedHEONN method for a classification task.

In this example, the clients and coordinator are created on the same machine.
In a real environment, the clients and also the coordinator can be created on
different machines. In that case, some communication mechanism must be
established between the clients and the coordinator to send the computations
performed by the clients.
"""

# Author: Oscar Fontenla-Romero <oscar.fontenla@udc.es>
# License: GPL-3.0-only

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from fedROLANN import FEDRolann, FEDRolann_Coordinator
from random import seed, shuffle
from time import time
import os
import struct
from tqdm import tqdm

# HYPERPARAMETERS
# Seed random numbers
seed(0)
# Number of clients
n_clients = 100
# Number of clients per group
n_groups = 1
# Encryption
enc = False
# Sparse matrices
spr = False
# Regularization
lam = 0.01
# IID
iid = False
# Activation function
f_act = "logs"

n_runs = 1


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


data_path = os.path.join("Data", "MNIST", "raw")
X_train_full, y_train_full = load_mnist(data_path, kind="train")
X_test, y_test = load_mnist(data_path, kind="t10k")

# Split the full training set into a smaller training set and a validation set (if needed)
train_X, test_X, train_t, test_t = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

# Data normalization (z-score): mean 0 and std 1
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

train_X = torch.from_numpy(train_X[:5000]).float()
test_X = torch.from_numpy(test_X).float()

train_t = torch.from_numpy(train_t[:5000])
test_t = torch.from_numpy(test_t)

# Number of training and test data
n = len(train_t)
ntest = len(test_t)

# Non-IID option
iid = False  # Change this to True for IID scenario
if not iid:
    # For non-IID scenario, sort training data by class
    sorted_indices = torch.argsort(train_t)
    train_data = train_X[sorted_indices]
    train_labels = train_t[sorted_indices]
    print("non-IID scenario")
else:
    ind_list = torch.randperm(n)
    train_data = train_X[ind_list]
    train_labels = train_t[ind_list]
    print("IID scenario")

# Number of classes
nclasses = len(torch.unique(train_labels))
print(f"There are {nclasses} classes")

# One-hot encoding
t_onehot = torch.nn.functional.one_hot(
    train_labels.long(), num_classes=nclasses
).float()
t_onehot = t_onehot * 0.9 + 0.05

# Create the coordinator
coordinator = FEDRolann_Coordinator(f=f_act, lamb=lam, num_classes=nclasses, sparse=spr)

# Create a list of clients and fit clients with their local data
start = time()
for _ in range(n_runs):
    lst_clients = []
    samples_per_client = n // n_clients
    print("samples_per_client", samples_per_client)
    for i in tqdm(range(n_clients), desc="Training clients"):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_data = train_data[start_idx:end_idx]
        client_labels = t_onehot[start_idx:end_idx]

        client = FEDRolann(activation=f_act, num_classes=nclasses)
        client.update_weights(client_data, client_labels)
        lst_clients.append(client)
end = time()

time_taken = end - start
avg_time_per_run = time_taken / n_runs

print(
    f"Training complete. Total time: {time_taken:.2f} seconds. "
    f"Average time per run: {avg_time_per_run:.2f} seconds over {n_runs} runs."
)


def get_prediction(ex_client, coord, testX, testT):
    # Send the weights of the aggregate model to example client
    ex_client.set_params(coord.send_weights())
    # Predictions for the test set using one client
    test_y = torch.argmax(ex_client(testX), dim=0)
    # Global MSE for the 3 outputs
    acc = 100 * accuracy_score(testT, test_y)
    return acc


# Case where the coordinator adds all clients in a single step
def global_fit(lst_clients, coord, testX, testT):
    # Create a list with the parameters (M, US) of all the clients (trained previously)
    M = []
    U = []
    S = []
    for client in lst_clients:
        M_c, U_c, S_c = client.get_params()
        M.append(M_c)
        U.append(U_c)
        S.append(S_c)

    M = torch.stack(M, dim=0)
    U = torch.stack(U, dim=0)
    S = torch.stack(S, dim=0)

    # The coordinator aggregates the information provided by the clients
    # to obtain the weights of the collaborative model
    coord.aggregate(M, U, S)

    # Global accuracy for the outputs
    acc = get_prediction(lst_clients[0], coord, testX, testT)
    return acc


# Create clients groups
def group_clients(lst_clients, n_groups):
    groups = []
    # Create a list with the groups of clients
    for i in range(0, len(lst_clients), n_groups):
        print(f"Grouping clients: {i}:{i + n_groups}")
        group = lst_clients[i : i + n_groups]
        groups.append(group)
    return groups


# Returns a list of all client parameters (M, US) in a group
def get_params_group(group):
    M_grp, U_grp, S_grp = [], [], []
    for _, client in enumerate(group):
        M_c, U_c, S_c = client.get_params()
        M_grp.append(M_c)
        U_grp.append(U_c)
        S_grp.append(S_c)

    M_grp = torch.stack(M_grp, dim=0)
    U_grp = torch.stack(U_grp, dim=0)
    S_grp = torch.stack(S_grp, dim=0)
    return M_grp, U_grp, S_grp


# Case in which an incremental aggregation is performed in the coordinator by client groups
def incremental_fit(lst_clients, coord, n_groups, testX, testT):
    debug = True
    shuffle(lst_clients)
    groups = group_clients(lst_clients, n_groups)
    for ig, group in enumerate(groups):
        M_grp, U_grp, S_grp = get_params_group(group=group)
        coord.aggregate_parcial(M_list=M_grp, U_list=U_grp, S_list=S_grp)
        if debug:
            coord.calculate_weights()
            print(
                f"\t***Test accuracy incremental (group {ig+1}): "
                f"{get_prediction(lst_clients[0], coord, testX, testT):0.8f}"
            )

    # Calculate optimal weights
    coord.calculate_weights()

    # Global accuracy for the outputs
    acc = get_prediction(lst_clients[0], coord, testX, testT)
    return acc


if __name__ == "__main__":
    # First case: aggregation of all clients in a single step in the coordinator
    mse_glb = global_fit(
        lst_clients=lst_clients, coord=coordinator, testX=test_X, testT=test_t
    )
    # Second case: aggregating all clients but adding them incrementally by client groups in the coordinator
    mse_inc = incremental_fit(
        lst_clients=lst_clients,
        n_groups=n_groups,
        coord=coordinator,
        testX=test_X,
        testT=test_t,
    )
    end_total = time()
    print(
        f"Finished total training, time taken {end_total - start} seconds, which is {(end_total-start)/n_runs} seconds per run"
    )
    print(f"Test accuracy global: {mse_glb:0.2f}")
    print(f"Test accuracy incremental: {mse_inc:0.2f}")
