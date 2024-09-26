# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:11:21 2024

@author: Oscar & Alejandro
"""

from time import time
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.WeightTracker import WeightTracker


class ROLANN_Incremental(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lamb: float = 0.01,
        activation: str = "logs",
        sparse: bool = False,
        dropout_rate: float = 0.0,
    ):
        super(ROLANN_Incremental, self).__init__()

        self.num_classes = num_classes
        self.old_num_classes = 0
        self.lamb = lamb  # Regularization hyperparameter

        if activation == "logs":  # Logistic activation functions
            self.f = torch.sigmoid
            self.finv = lambda x: torch.log(x / (1 - x))
            self.fderiv = lambda x: x * (1 - x)
        elif activation == "rel":  # ReLU activation functions
            self.f = F.relu
            self.finv = lambda x: torch.log(x)
            self.fderiv = lambda x: (x > 0).float()
        elif activation == "lin":  # Linear activation functions
            self.f = lambda x: x
            self.finv = lambda x: x
            self.fderiv = lambda x: torch.ones_like(x)

        self.w = []

        self.m = None
        self.u = None
        self.s = None

        self.mg = []
        self.ug = []
        self.sg = []

        self.sparse = sparse
        self.dropout = nn.Dropout(dropout_rate)

        self.weight_tracker = WeightTracker()

    def add_num_classes(self, num_classes):
        self.old_num_classes = self.num_classes
        self.num_classes += num_classes

    def update_weights(self, X: Tensor, d: Tensor, classes: Tensor) -> Tensor:
        results = [self._update_weights(X, d[:, i]) for i in classes]

        ml, ul, sl = zip(*results)

        self.m = torch.stack(ml, dim=0)
        self.u = torch.stack(ul, dim=0)
        self.s = torch.stack(sl, dim=0)

    def _update_weights(self, X: Tensor, d: Tensor) -> Tensor:

        X = X.T
        n = X.size(1)  # Number of data points (n)

        # The bias is included as the first input (first row)
        ones = torch.ones((1, n), device=X.device)

        xp = torch.cat((ones, X), dim=0)

        # Inverse of the neural function
        f_d = self.finv(d)

        # Derivative of the neural function
        derf = self.fderiv(f_d)

        if self.sparse:
            F_sparse = torch.diag(derf)

            H = torch.matmul(xp, F_sparse)

            U, S, _ = torch.linalg.svd(H, full_matrices=False)

            M = torch.matmul(
                xp, torch.matmul(F_sparse, torch.matmul(F_sparse, f_d.T))
            ).flatten()
        else:
            # Diagonal matrix
            F = torch.diag(derf)

            H = torch.matmul(xp, F)

            U, S, _ = torch.linalg.svd(H, full_matrices=False)

            M = torch.matmul(xp, torch.matmul(F, torch.matmul(F, f_d)))

        return M, U, S

    def reset(self) -> None:
        self.ug = []
        self.sg = []
        self.mg = []
        self.w = None

    def forward(self, X: Tensor) -> Tensor:
        X = X.T
        n = X.size(1)

        n_outputs = len(self.w)

        # Neural Network Simulation
        ones = torch.ones((1, n), device=X.device)
        xp = torch.cat((ones, X), dim=0)

        y_hat = torch.empty((n_outputs, n), device=X.device)

        for i in range(n_outputs):
            w_tmp = self.w[i].permute(
                *torch.arange(self.w[i].ndim - 1, -1, -1)
            )  # Trasposing

            y_hat[i] = self.f(torch.matmul(w_tmp, self.dropout(xp)))

        return torch.transpose(y_hat, 0, 1)

    def _aggregate_parcial(self, classes: Tensor) -> None:
        for i, c in enumerate(classes):
            if c >= len(self.mg):
                # Initialization using the first element of the list
                M = self.m[i]
                U = self.u[i]
                S = self.s[i]

                self.mg.append(M)
                self.ug.append(U)
                self.sg.append(S)

            else:
                M = self.mg[c]
                m_k = self.m[i]
                s_k = self.s[i]
                u_k = self.u[i]

                US = torch.matmul(self.ug[c], torch.diag(self.sg[c]))

                # Aggregation of M and US
                M = M + m_k
                us_k = torch.matmul(u_k, torch.diag(s_k))
                concatenated = torch.cat((us_k, US), dim=1)
                U, S, _ = torch.linalg.svd(concatenated, full_matrices=False)

                self.mg[c] = M
                self.ug[c] = U
                self.sg[c] = S

    def _calculate_weights(self, classes: Tensor) -> None:
        if not self.mg or not self.ug or not self.sg:
            return None

        for c in classes:
            M = self.mg[c]
            U = self.ug[c]
            S = self.sg[c]

            if self.sparse:
                I_ones = torch.ones(S.size())
                I_ones_size = list(I_ones.shape)[0]
                I_sparse = torch.sparse.spdiags(
                    I_ones,
                    torch.tensor(0),
                    (I_ones_size, I_ones_size),
                    layout=torch.sparse_csr,
                )
                S_size = list(S.shape)[0]
                S_sparse = torch.sparse.spdiags(
                    S, torch.tensor(0), (S_size, S_size), layout=torch.sparse_csr
                )

                aux = (
                    S_sparse.to_dense() * S_sparse.to_dense()
                    + self.lamb * I_sparse.to_dense()
                )
                # Optimal weights: the order of the matrix and vector multiplications has been done to optimize the speed
                w = torch.matmul(
                    U, torch.matmul(torch.linalg.pinv(aux), torch.matmul(U.T, M))
                )
            else:
                diag_elements = 1 / (
                    S * S + self.lamb * torch.ones_like(S, device=S.device)
                )
                diag_matrix = torch.diag(diag_elements)
                # Optimal weights: the order of the matrix and vector multiplications has been done to optimize the speed
                w = torch.matmul(U, torch.matmul(diag_matrix, torch.matmul(U.T, M)))

            if c >= len(self.w):
                # Append optimal weights
                self.w.append(w)
            else:
                self.w[c] = w

            self.weight_tracker.update(c.item(), w)

    # Add a method to visualize weights
    def visualize_weights(self):
        self.weight_tracker.plot_weight_statistics()
        # self.weight_tracker.plot_weight_distribution()

    def aggregate_update(self, X: Tensor, d: Tensor):
        unique_classes = torch.argmax(d, dim=1).unique()

        self.update_weights(X, d, unique_classes)  # Se calculan las nuevas M y US
        self._aggregate_parcial(
            unique_classes
        )  # Se agrega nuevas M y US a antiguas (globales)
        self._calculate_weights(unique_classes)  # Se calcula los pesos con las nuevas
