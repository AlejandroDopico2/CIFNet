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

    classifier_type = config["model"].get("classifier", "rolann")
    classifier_kwargs = config["model"].get("classifier_params", {}).copy()
    rolann_cfg = config.get("rolann", {})

    if config["model"]["backbone"]:
        backbone = get_backbone_class(
            "models.backbone", config["model"]["backbone"] + "Backbone"
        )
    else:
        backbone = None

    model = CIFNet(
        num_classes=0,
        activation=classifier_kwargs.get("activation", "logs"),
        lamb=rolann_cfg.get("rolann_lamb", 0.01),
        normalize=classifier_kwargs.get("normalize", True),
        sparse=rolann_cfg.get("sparse", False),
        pretrained=config["model"]["pretrained"],
        backbone=backbone,
        in_channels=in_channels,
        device=config["device"],
        freeze_mode=config["model"].get("freeze_mode", "all"),
        classifier_type=classifier_type,
        classifier_kwargs=classifier_kwargs,
    ).to(config["device"])

    return model
