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
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from fedROLANN import FEDRolann, FEDRolann_Coordinator
from random import seed, shuffle

from time import time

# HYPERPARAMETERS
# Seed random numbers
seed(0)
# Number of clients
n_clients = 200
# Number of clients per group
n_groups = 20
# Encryption
enc = False
# Sparse matrices
spr = False
# Regularization
lam = 0.01
# IID
iid = True
# Activation function
f_act = "logs"

n_runs = 1

# The data set is loaded (Dry Bean Dataset)
# Source: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset
# Article: https://www.sciencedirect.com/science/article/pii/S0168169919311573?via%3Dihub
Data = pd.read_excel("./data/Dry_Bean_Dataset.xlsx", sheet_name="Dry_Beans_Dataset")
Data["Class"] = Data["Class"].map(
    {
        "BARBUNYA": 0,
        "BOMBAY": 1,
        "CALI": 2,
        "DERMASON": 3,
        "HOROZ": 4,
        "SEKER": 5,
        "SIRA": 6,
    }
)
Inputs = Data.iloc[:, :-1].to_numpy()
Labels = Data.iloc[:, -1].to_numpy()
train_X, test_X, train_t, test_t = train_test_split(
    Inputs, Labels, test_size=0.3, random_state=42
)

# Data normalization (z-score): mean 0 and std 1
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

train_X = torch.from_numpy(train_X).float()
test_X = torch.from_numpy(test_X).float()

train_t = torch.from_numpy(train_t)
test_t = torch.from_numpy(test_t)

# Number of training and test data
n = len(train_t)
ntest = len(test_t)

# Non-IID option: Sort training data by class
if not iid:
    ind = torch.argsort(train_t)
    train_t = train_t[ind]
    train_X = train_X[:, ind]
    print("non-IID scenario")
else:
    ind_list = torch.randperm(n)
    train_X = train_X[ind_list, :]
    train_t = train_t[ind_list]
    print("IID scenario")

# Number of classes
nclasses = len(torch.unique(train_t))

print(f"There are {nclasses} classes")

t_onehot = torch.nn.functional.one_hot(train_t)

t_onehot = t_onehot * 0.9 + 0.05

# Create the coordinator
coordinator = FEDRolann_Coordinator(f=f_act, lamb=lam, num_classes=nclasses, sparse=spr)

# Create a list of clients and fit clients with their local data
start = time()
for _ in range(n_runs):
    lst_clients = []
    for i in range(0, n_clients):
        rang = range(
            int(i * n / n_clients), int(i * n / n_clients) + int(n / n_clients)
        )
        client = FEDRolann(activation=f_act, num_classes=nclasses)
        # print('Training client:', i+1, 'of', n_clients, '(', min(rang), '-', max(rang), ') - Classes:', np.unique(train_t[rang]))
        client.update_weights(train_X[rang, :], t_onehot[rang, :])
        lst_clients.append(client)

end = time()

print(
    f"Finished training client, time taken {end - start} seconds, whic is {(end-start)/n_runs} seconds per run"
)


def get_prediction(ex_client, coord, testX, testT):
    # Send the weights of the aggregate model to example client
    ex_client.set_params(coord.send_weights())
    # Predictions for the test set using one client
    test_y = torch.argmax(ex_client(testX), dim=0)
    # Global MSE for the 3 outputs
    acc = 100 * accuracy_score(test_t, test_y)
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
        coordinator.aggregate_parcial(M_list=M_grp, U_list=U_grp, S_list=S_grp)
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
