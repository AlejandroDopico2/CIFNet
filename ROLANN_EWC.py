# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:11:21 2024

@author: Oscar & Alejandro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RolannEWC(nn.Module):
    def __init__(self, alpha: int = 0, activation: str = "logs"):
        super(RolannEWC, self).__init__()
        self.alpha = alpha  # Regularization hyperparameter

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
        self.fisher = None

    def update_weights(self, X: Tensor, d: Tensor) -> Tensor:
        X = X.T
        n = X.size(1)

        # The bias is included as the first input (first row)
        ones = torch.ones((1, n), device=X.device)

        Xp = torch.cat((ones, X), dim=0)

        # Inverse of the neural function
        f_d = self.finv(d)

        # Derivative of the neural function
        derf = self.fderiv(f_d)

        # Diagonal matrix
        F = torch.diag(derf)

        XFF = torch.matmul(torch.matmul(Xp, F), F)
        fisher = torch.matmul(XFF, Xp.T)

        # Solution
        if self.w is not None and self.fisher is not None:
            L = fisher + self.alpha * self.fisher
            R = torch.matmul(XFF, f_d) + self.alpha * torch.matmul(self.fisher, self.w)
        else:
            L = fisher
            R = torch.matmul(XFF, f_d)

        w = torch.matmul(torch.linalg.pinv(L), R)

        self.w = w
        self.fisher = fisher

    def forward(self, X: Tensor, w: Tensor = None) -> Tensor:
        X = X.T
        n = X.size(1)

        if w is not None:
            self.w = w

        # Neural Network Simulation
        ones = torch.ones((1, n), device=X.device)
        Xp = torch.cat((ones, X), dim=0)

        self.w = self.w.permute(*torch.arange(self.w.ndim - 1, -1, -1))
        x = torch.matmul(self.w, Xp)

        return self.f(x)

    def reset(self) -> None:
        self.w = None
        self.fisher = None
