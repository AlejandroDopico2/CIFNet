# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:11:21 2024

@author: Oscar & Alejandro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as parallel
from torch import Tensor


class ROLANN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lamb: float = 0.01,
        activation: str = "logs",
        sparse: bool = False,
        dropout_rate: float = 0.0,
    ):
        super(ROLANN, self).__init__()

        self.num_classes = num_classes
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

        self.w = None

        self.m = None
        self.u = None
        self.s = None

        self.mg = None
        self.ug = None
        self.sg = None

        self.sparse = sparse
        self.dropout = nn.Dropout(dropout_rate)

    def update_weights(self, X: Tensor, d: Tensor) -> Tensor:
        ml_list = []
        ul_list = []
        sl_list = []

        update_funcs = [lambda x=X, di=d[:, i]: self._update_weights(x, di) for i in range(self.num_classes)]

        results = parallel.parallel_apply(update_funcs, [tuple() for _ in range(self.num_classes)])

        ml_list, ul_list, sl_list = zip(*results)

        self.m = torch.stack(ml_list, dim=0)
        self.u = torch.stack(ul_list, dim=0)
        self.s = torch.stack(sl_list, dim=0)

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
            # F = torch.diag(derf)

            xp_derf = xp * derf

            U, S, _ = torch.linalg.svd(xp_derf, full_matrices=False)

            M = torch.matmul(xp_derf, derf * f_d.T).flatten()

        return M, U, S

    def reset(self) -> None:
        self.ug = None
        self.sg = None
        self.mg = None
        self.w = None

    def forward(self, X: Tensor) -> Tensor:
        X = X.T
        n = X.size(1)

        n_outputs = len(self.w)

        # Neural Network Simulation
        ones = torch.ones((1, n), device=X.device)
        xp = torch.cat((ones, X), dim=0)

        # Stack weights for parallel computation
        stacked_w = torch.stack(self.w)  # Shape: (n_outputs, dim_1, dim_2, ..., input_dim + 1)
        
        # Apply dropout and expand xp for matrix multiplication with each weight tensor
        xp = self.dropout(xp)  # Dropout can remain outside the loop, applied once
        xp_expanded = xp.unsqueeze(0).expand(n_outputs, *xp.size())  # Shape: (n_outputs, input_dim + 1, n)

        transposed_w = stacked_w.permute(0, 2, 1)  # Transpose the last two dimensions

        # Compute the outputs using a batched matrix multiplication
        y_hat = self.f(torch.matmul(transposed_w, xp_expanded))  # Shape: (n_outputs, n)

        return torch.transpose(y_hat.squeeze(), 0, 1)


    def get_params(self):
        return self.m, self.us

    def set_params(self, w):
        self.w = w

    def _aggregate_parcial(self) -> None:

        if self.mg is None:
            self.mg = self.m
            self.ug = self.u
            self.sg = self.s
        else:
            device = self.m.device
        
            M = self.mg
            m_k = self.m
            u_k = self.u
            s_k = self.s

            US = self.ug * self.sg.unsqueeze(1)
            us_k = u_k * s_k.unsqueeze(1)

            # Aggregation of M and US
            M += m_k

            # Efficient concatenation
            concat_dim = us_k.shape[2] + US.shape[2]
            concatenated = torch.zeros((self.num_classes, US.shape[1], concat_dim), device=device)
            concatenated[:, :, :us_k.shape[2]] = us_k
            concatenated[:, :, us_k.shape[2]:] = US

            # Perform SVD and update
            U, S, _ = torch.linalg.svd(concatenated, full_matrices=False)

            self.mg = M
            self.ug = U
            self.sg = S

    # @torch.jit.script
    def _calculate_weights_sparse(self, M: torch.Tensor, U: torch.Tensor, S: torch.Tensor, lamb: float) -> torch.Tensor:
        I_ones = torch.ones_like(S)
        device = S.device
        S_sparse = torch.sparse_csr_tensor(
            torch.arange(S.size(0) + 1, device=device),
            torch.arange(S.size(0), device=device),
            S,
            size=(S.size(0), S.size(0))
        )
        I_sparse = torch.sparse_csr_tensor(
            torch.arange(S.size(0) + 1, device=device),
            torch.arange(S.size(0), device=device),
            I_ones,
            size=(S.size(0), S.size(0))
        )
        
        aux = S_sparse.to_dense().pow(2) + lamb * I_sparse.to_dense()
        return torch.chain_matmul(U, torch.linalg.pinv(aux), U.T, torch.unsqueeze(M, dim = 1))

    # @torch.jit.script
    def _calculate_weights_dense(self, M: torch.Tensor, U: torch.Tensor, S: torch.Tensor, lamb: float) -> torch.Tensor:
        diag_elements = 1 / (S.pow(2) + lamb)
        return torch.chain_matmul(U, torch.diag(diag_elements), U.T, torch.unsqueeze(M, dim = 1))

    def _calculate_weights(self) -> None:
        if self.mg is None or self.ug is None or self.sg is None:
            self.w = None
            return
        
        self.w = []
        
        for c in range(self.num_classes):
            M, U, S = self.mg[c], self.ug[c], self.sg[c]

            if self.sparse:
                w = self._calculate_weights_sparse(M, U, S, self.lamb)
            else:
                w = self._calculate_weights_dense(M, U, S, self.lamb)

            self.w.append(w)


    def aggregate_update(self, X: Tensor, d: Tensor):
        self.update_weights(X, d)  # Se calculan las nuevas M y US
        self._aggregate_parcial()  # Se agrega nuevas M y US a antiguas (globales)
        self._calculate_weights()  # Se calcula los pesos con las nuevas
