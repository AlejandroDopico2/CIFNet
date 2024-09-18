from models.ROLANN_incremental import ROLANN_Incremental
from models.backbone import Backbone
from models.ROLANN import ROLANN as ROLANN
from models.ROLANN_optim import ROLANN as ROLANN_optim
import torch.nn as nn
import torch
from typing import Optional


class RolanNET(nn.Module):
    def __init__(
        self,
        num_classes: int,
        activation: str = "logs",
        lamb: float = 0.01,
        pretrained: bool = True,
        backbone: Optional[Backbone] = None,
        in_channels: int = 3,
        sparse: bool = False,
        device: str = "cuda",
        dropout_rate: float = 0.0,
        optim: bool = False,
        freeze: bool = False,
        incremental: bool = False
    ) -> None:
        super(RolanNET, self).__init__()

        self.device = device

        if backbone is not None:
            self.backbone = backbone(pretrained).to(self.device)
            self.backbone.set_input_channels(in_channels)
            self.freeze_backbone(freeze)
        else:
            self.backbone = None

        if optim:
            self.rolann = ROLANN_optim(
                num_classes, activation=activation, lamb=lamb, sparse=sparse, dropout_rate = dropout_rate,
            ).to(self.device)
        else:
            self.rolann = ROLANN(
                num_classes, activation=activation, lamb=lamb, sparse=sparse, dropout_rate = dropout_rate,
            ).to(self.device)

        if incremental:
            self.rolann = ROLANN_Incremental(
                num_classes, activation=activation, lamb=lamb, sparse=sparse, dropout_rate = dropout_rate,
            ).to(self.device)
        else:
            self.rolann = ROLANN(
                num_classes, activation=activation, lamb=lamb, sparse=sparse, dropout_rate = dropout_rate,
            ).to(self.device)

    def freeze_backbone(self, freeze: bool = False) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone:
            x = x.to(self.device)
            x = self.backbone(x).squeeze()

        x = x.to(self.device)
        x = self.rolann(x)

        return x

    @torch.no_grad
    def update_rolann(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        if self.backbone:
            x = x.to(self.device)
            x = self.backbone(x).squeeze()

        x = x.to(self.device)

        self.rolann.aggregate_update(x, labels.to(self.device))

    def reset_rolann(self) -> None:
        self.rolann.reset()
