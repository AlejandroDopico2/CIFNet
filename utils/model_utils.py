import torch.nn as nn
import importlib
from typing import Any, Dict, Type
from models.CIFNet import CIFNet
from models.backbone import Backbone


def get_backbone_class(module_name: str, class_name: str) -> Type[Backbone]:
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if issubclass(cls, Backbone):
        return cls
    else:
        raise ValueError(f"Class {class_name} is not a subclass of Backbone")


def build_incremental_model(config: Dict[str, Any]) -> nn.Module:
    in_channels = 1 if config["dataset"]["name"] == "MNIST" else 3

    if config["model"]["backbone"]:
        backbone = get_backbone_class(
            "models.backbone", config["model"]["backbone"] + "Backbone"
        )
    else:
        backbone = None

    model = CIFNet(
        num_classes=0,
        activation="logs",
        lamb=config["rolann"]["rolann_lamb"],
        pretrained=config["model"]["pretrained"],
        backbone=backbone,
        in_channels=in_channels,
        sparse=config["rolann"]["sparse"],
        dropout_rate=config["rolann"]["dropout_rate"],
        device=config["device"],
        freeze_mode=config["model"]["freeze_mode"],
    ).to(config["device"])

    return model
