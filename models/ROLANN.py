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

        if self.sparse:
            print("Warning: Sparse mode is not optimized for GPU parallelization and has been disabled in the parallel implementation of _calculate_weights.")

    def add_num_classes(self, num_classes):
        self.num_classes += num_classes

    def update_weights(self, X: Tensor, d: Tensor, classes: Tensor) -> None:
        """
        Computes M, U, and S for a batch of classes in parallel, removing the original loop.
        """
        num_samples = X.size(0)

        ones = torch.ones((num_samples, 1), device=X.device) * 0.1
        xp = torch.cat((ones, X), dim=1).T

        d_t = d.T
        f_d = self.finv(d_t)
        derf = self.fderiv(f_d) # Shape: (num_classes_in_update, num_samples)

        F = torch.diag_embed(derf)

        # === PARALLELIZED SVD COMPUTATION ===
        H = torch.matmul(xp.unsqueeze(0), F)
        
        U, S, _ = torch.linalg.svd(H, full_matrices=False)

        # === PARALLELIZED M COMPUTATION ===
        f_d_vec = f_d.unsqueeze(-1)
        M = xp.unsqueeze(0) @ F @ (F @ f_d_vec)
        
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

        W = torch.stack(self.w, dim=0)

        y_hat = self.f(torch.matmul(W, xp))

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

        s_squared = S_batch ** 2
        avg_energy = torch.mean(s_squared, dim=1, keepdim=True)
        adaptive_lamb = self.lamb * avg_energy + 1e-7

        inv_diag_matrix = torch.diag_embed(1.0 / (s_squared + adaptive_lamb))

        ut_m = U_batch.transpose(-2, -1) @ M_batch.unsqueeze(-1)
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

        # print(f"CLASSES TO PROCESS: {classes_to_process}")

        d_filtered = d[:, classes_to_process]
        
        self.update_weights(X, d_filtered, classes_to_process)
        self._aggregate_parcial(classes_to_process)
        self._calculate_weights(classes_to_process)