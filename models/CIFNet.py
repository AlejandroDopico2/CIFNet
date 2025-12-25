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
        freeze_mode: str = "all",
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
            self.freeze_backbone(freeze_mode)
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

        self.register_buffer("running_mean", torch.zeros(1, 512)) # feature_dim=768 for ViT
        self.navg = 0


    def freeze_backbone(self, freeze_mode: str) -> None:
        if freeze_mode == "none":
            # No freezing
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif freeze_mode == "all":
            # Freeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_mode == "partial":
            # Freeze all layers except the last few
            total_layers = len(list(self.backbone.children()))
            for i, child in enumerate(self.backbone.children()):
                if i < total_layers - 2:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True
        else:
            raise ValueError(
                f"Invalid freeze_mode: {freeze_mode}. Choose 'none', 'all', or 'partial'."
            )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization and centering. 
        Crucial for ViT backbones to combat feature anisotropy.
        """
        # 1. L2 Normalization (Project to hypersphere)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        # 2. Centering (Subtract running mean)
        # Note: We update the mean only during training/aggregation
        if self.training:
            n = x.size(0)
            # Ensure running_mean is on the correct device
            if self.running_mean.device != x.device:
                self.running_mean = self.running_mean.to(x.device)
                
            batch_mean = x.mean(dim=0, keepdim=True)
            self.running_mean = (self.navg * self.running_mean + n * batch_mean) / (self.navg + n)
            self.navg += n

        return x - self.running_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone:
            x = x.to(self.device)
            x = self.backbone(x)
            x = x.flatten(start_dim=1)
        
        x = self.preprocess(x)

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
        x = self.preprocess(x)
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
