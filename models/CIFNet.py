from models.ROLANN import ROLANN
from models.classifiers import (
    RolannClassifier,
    TorchLinearClassifier,
    SklearnLogRegClassifier,
    NCMClassifier,
)
from models.backbone import Backbone
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Dict


class CIFNet(nn.Module):
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
        classifier_type: str = "rolann",
        classifier_kwargs: Optional[Dict] = None,
        normalize: bool = True,
    ) -> None:
        super(CIFNet, self).__init__()

        self.device = device
        self.classifier_type = classifier_type.lower()
        self.classifier_kwargs = classifier_kwargs or {}
        self.normalize = normalize
        if backbone is not None:
            self.backbone = backbone(pretrained).to(self.device)
            self.backbone.set_input_channels(in_channels)
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self.backbone = None

        self.classifier = self._init_classifier(
            classifier_type=self.classifier_type,
            num_classes=num_classes,
            activation=activation,
            lamb=lamb,
            sparse=sparse,
        )
        # Backwards-compatibility alias
        self.rolann = self.classifier


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone:
            x = x.to(self.device)
            x = self.backbone(x)
            x = x.flatten(start_dim=1)
        
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        x = x.to(self.device)
        x = self.classifier(x)

        return x

    def update_classifier(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        classes: Optional[int] = None,
        is_embedding: bool = False,
    ) -> None:
        if self.backbone and not is_embedding:
            x = x.to(self.device)
            x = self.backbone(x).flatten(start_dim=1)
        
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        x = x.to(self.device)

        self.classifier.aggregate_update(x, labels.to(self.device), classes=classes)

    # Backwards compatibility for existing training code
    def update_rolann(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        classes: Optional[int] = None,
        is_embedding: bool = False,
    ) -> None:
        self.update_classifier(x, labels, classes=classes, is_embedding=is_embedding)

    def add_num_classes(self, num_classes: int) -> None:
        self.classifier.add_num_classes(num_classes)
        # Keep alias in sync
        self.rolann = self.classifier

    def _init_classifier(
        self,
        classifier_type: str,
        num_classes: int,
        activation: str,
        lamb: float,
        sparse: bool,
    ):
        ctype = classifier_type.lower()
        if ctype == "rolann":
            return RolannClassifier(
                num_classes,
                activation=activation,
                lamb=lamb,
                sparse=sparse,
            ).to(self.device)
        if ctype == "linear":
            return TorchLinearClassifier(
                num_classes,
                lr=self.classifier_kwargs.get("lr", 0.01),
                weight_decay=self.classifier_kwargs.get("weight_decay", 0.0),
                device=self.device,
            )
        if ctype in {"logreg", "logistic", "logistic_regression"}:
            return SklearnLogRegClassifier(
                num_classes,
                alpha=self.classifier_kwargs.get("alpha", 0.0001),
                lr=self.classifier_kwargs.get("lr", 0.01),
                device=self.device,
            )
        if ctype in {"ncm", "prototype", "proto"}:
            return NCMClassifier(num_classes, device=self.device)

        raise ValueError(f"Unsupported classifier type: {classifier_type}")
