# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:11:21 2024

@author: Oscar & Alejandro

Parallelized version by Gemini
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.lamb = lamb
        self.sparse = sparse

        if activation == "logs":
            self.f = torch.sigmoid
            self.finv = lambda x: torch.log(x / (1 - x))
            self.fderiv = lambda x: x * (1 - x)
        elif activation == "rel":
            self.f = F.relu
            self.finv = lambda x: torch.log(x.clamp(min=1e-8))
            self.fderiv = lambda x: (x > 0).float()
        elif activation == "lin":
            self.f = lambda x: x
            self.finv = lambda x: x
            self.fderiv = lambda x: torch.ones_like(x)

        # Weights are stored as a list of tensors, one per class
        self.w: List[Tensor] = []

        # Intermediate tensors for updates
        self.m: Optional[Tensor] = None
        self.u: Optional[Tensor] = None
        self.s: Optional[Tensor] = None

        # Global aggregated tensors, stored as a list
        self.mg: List[Tensor] = []
        self.ug: List[Tensor] = []
        self.sg: List[Tensor] = []

        self.dropout = nn.Dropout(dropout_rate)
        
        if self.sparse:
            print("Warning: Sparse mode is not optimized for GPU parallelization and has been disabled in the parallel implementation of _calculate_weights.")

    def add_num_classes(self, num_classes):
        self.num_classes += num_classes

    def update_weights(self, X: Tensor, d: Tensor, classes: Tensor) -> None:
        """
        Computes M, U, and S for a batch of classes in parallel, removing the original loop.
        """
        # d shape: (batch_size, num_classes_in_update)
        # X shape: (batch_size, num_features)
        num_samples = X.size(0)

        # The bias is included as the first feature. xp shape: (num_features + 1, num_samples)
        ones = torch.ones((num_samples, 1), device=X.device)
        xp = torch.cat((ones, X), dim=1).T

        # Inverse and derivative calculations are now batched over classes
        # Transpose d to have classes as the batch dimension: (num_classes_in_update, num_samples)
        d_t = d.T
        f_d = self.finv(d_t)
        derf = self.fderiv(f_d) # Shape: (num_classes_in_update, num_samples)

        # Create a batched diagonal matrix F. Shape: (num_classes_in_update, num_samples, num_samples)
        F = torch.diag_embed(derf)

        # === PARALLELIZED SVD COMPUTATION ===
        # H shape: (num_classes_in_update, num_features + 1, num_samples)
        # xp (features, samples) is broadcasted across the class dimension of F
        H = torch.matmul(xp.unsqueeze(0), F)
        
        U, S, _ = torch.linalg.svd(H, full_matrices=False)
        # U shape: (num_classes_in_update, num_features + 1, k)
        # S shape: (num_classes_in_update, k) where k=min(features+1, samples)

        # === PARALLELIZED M COMPUTATION ===
        # Reshape f_d for batched matmul: (num_classes_in_update, num_samples, 1)
        f_d_vec = f_d.unsqueeze(-1)
        # M = xp @ F @ F @ f_d
        M = xp.unsqueeze(0) @ F @ (F @ f_d_vec) # Shape: (num_classes_in_update, num_features + 1, 1)
        
        self.m = M.squeeze(-1)
        self.u = U
        self.s = S


    def forward(self, X: Tensor) -> Tensor:
        """
        Parallelized forward pass.
        """
        if not self.w:
            return torch.zeros((X.size(0), self.num_classes), device=X.device)

        num_samples = X.size(0)

        # Add bias term to X. xp shape: (num_features + 1, num_samples)
        ones = torch.ones((num_samples, 1), device=X.device)
        xp = torch.cat((ones, X), dim=1).T

        # === PARALLELIZED PREDICTION ===
        # W shape: (num_classes, num_features + 1)
        W = torch.stack(self.w, dim=0)

        # (num_classes, features) @ (features, samples) -> (num_classes, samples)
        y_hat = self.f(torch.matmul(W, self.dropout(xp)))

        return y_hat.T


    def _aggregate_parcial(self, classes: Tensor) -> None:
        """
        This part remains a loop because it's a stateful, sequential update.
        However, the M, U, S it uses are now computed in a parallel batch.
        """
        for i, c in enumerate(classes):
            m_k, u_k, s_k = self.m[i], self.u[i], self.s[i]

            if c >= len(self.mg):
                self.mg.append(m_k)
                self.ug.append(u_k)
                self.sg.append(s_k)
            else:
                M_g, U_g, S_g = self.mg[c], self.ug[c], self.sg[c]

                # Aggregate M
                M_new = M_g + m_k

                # Aggregate US by concatenating and re-running SVD
                US_g = U_g @ torch.diag(S_g)
                us_k = u_k @ torch.diag(s_k)
                
                concatenated = torch.cat((US_g, us_k), dim=1)
                U_new, S_new, _ = torch.linalg.svd(concatenated, full_matrices=False)

                # Update global components
                self.mg[c] = M_new
                self.ug[c] = U_new
                self.sg[c] = S_new


    def _calculate_weights(self, classes: List) -> None:
        """
        Calculates weights in a parallel batch for all specified classes.
        """
        if not self.mg:
            return
        
        M_list = [self.mg[c] for c in classes]
        U_list = [self.ug[c] for c in classes]
        S_list = [self.sg[c] for c in classes]

        max_k = max(u.shape[1] for u in U_list)

        padded_U_list = []
        padded_S_list = []
        for u, s in zip(U_list, S_list):
            # Calculate how much padding is needed for the current tensor
            k_diff = max_k - u.shape[1]
            
            # Pad U on the right side of the columns dimension
            # (pad_left, pad_right, pad_top, pad_bottom)
            padded_u = F.pad(u, (0, k_diff, 0, 0), "constant", 0)
            padded_U_list.append(padded_u)

            # Pad S at the end of the vector
            padded_s = F.pad(s, (0, k_diff), "constant", 0)
            padded_S_list.append(padded_s)

        # Gather the components for the specified classes
        M_batch = torch.stack(M_list, dim=0)
        U_batch = torch.stack(padded_U_list, dim=0)
        S_batch = torch.stack(padded_S_list, dim=0)

        # === PARALLELIZED WEIGHT CALCULATION ===
        # M_batch shape: (num_classes, features), needs to be (num_classes, features, 1) for bmm
        M_vec = M_batch.unsqueeze(-1)

        s_squared = S_batch * S_batch
        denominator = s_squared + self.lamb
        
        diag_elements = 1.0 / denominator
        inv_diag_matrix = torch.diag_embed(diag_elements)

        # w = U @ inv(S^2 + lambda) @ U.T @ M
        ut_m = U_batch.transpose(-2, -1) @ M_vec
        w_batch = U_batch @ (inv_diag_matrix @ ut_m)
        w_batch = w_batch.squeeze(-1)

        for i, c in enumerate(classes):
            if c >= len(self.w):
                self.w.extend([None] * (c + 1 - len(self.w)))
            self.w[c] = w_batch[i]


    def aggregate_update(self, X: Tensor, d: Tensor, classes: Optional[Tensor]) -> None:
        """
        The main training step, now using parallelized sub-routines.
        """
        if classes is None:
            classes_to_process = torch.arange(self.num_classes, device=d.device)
        else:
            classes_to_process = classes

        d_filtered = d[:, classes_to_process]
        
        self.update_weights(X, d_filtered, classes_to_process)
        self._aggregate_parcial(classes_to_process)
        self._calculate_weights(classes_to_process)