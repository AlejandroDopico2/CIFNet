import torch.nn as nn
import importlib
from typing import Any, Dict, Type
from models.RolanNET import RolanNET
from models.Backbone import Backbone


def get_backbone_class(module_name: str, class_name: str) -> Type[Backbone]:
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if issubclass(cls, Backbone):
        return cls
    else:
        raise ValueError(f"Class {class_name} is not a subclass of Backbone")


def build_model(config: Dict[str, Any]) -> nn.Module:
    in_channels = 1 if config["dataset"] == "MNIST" else 3

    if config["backbone"]:
        backbone = get_backbone_class(
            "models.Backbone", config["backbone"] + "Backbone"
        )
    else:
        backbone = None

    model = RolanNET(
        num_classes=config["num_classes"],
        activation="logs",
        lamb=config["rolann_lamb"],
        pretrained=config["pretrained"],
        backbone=backbone,
        in_channels=in_channels,
        sparse=config["sparse"],
        dropout_rate=config["dropout_rate"],
        device=config["device"],
    ).to(config["device"])

    return model


def build_incremental_model(config: Dict[str, Any]) -> nn.Module:
    in_channels = 1 if config["dataset"] == "MNIST" else 3

    if config["backbone"]:
        backbone = get_backbone_class(
            "models.Backbone", config["backbone"] + "Backbone"
        )
    else:
        backbone = None

    model = RolanNET(
        num_classes=0,
        activation="logs",
        lamb=config["rolann_lamb"],
        pretrained=config["pretrained"],
        backbone=backbone,
        in_channels=in_channels,
        sparse=config["sparse"],
        dropout_rate=config["dropout_rate"],
        device=config["device"],
        incremental=True,
        freeze=config["freeze"]
    ).to(config["device"])

    return model
