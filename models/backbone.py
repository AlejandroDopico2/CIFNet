
from abc import ABC, abstractmethod
import os

import torch.nn as nn
from torch import Tensor
from torchvision import models
import torch
import timm


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "teacher" in checkpoint:
            return checkpoint["teacher"]
        if "student" in checkpoint:
            return checkpoint["student"]
    return checkpoint


def _load_checkpoint_from_path_or_url(path_or_url: str):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        checkpoint = torch.hub.load_state_dict_from_url(path_or_url, map_location="cpu")
    else:
        checkpoint = torch.load(path_or_url, map_location="cpu")
    return _extract_state_dict(checkpoint)

class Backbone(ABC, nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.backbone_type = "cnn"

    @abstractmethod
    def set_input_channels(self, channels: int):
        pass

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model(x)

class ViTB16Backbone(Backbone):
    def __init__(self, pretrained: bool = True):
        super(ViTB16Backbone, self).__init__()
        self.backbone_type = "vit"
        self.model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=True
        )
        self.model.head = nn.Identity()
        self.model.eval()
        
    def set_input_channels(self, channels: int):
        pass

class ResNet18Backbone(Backbone):
    def __init__(self, pretrained: bool = True):
        super(ResNet18Backbone, self).__init__()
        self.model = (
            models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            if pretrained
            else models.resnet18()
        )
        self.model.fc = nn.Identity()
        
    def set_input_channels(self, channels: int):
        if channels == 1:
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

class ResNet34Backbone(Backbone):
    def __init__(self, pretrained: bool = True):
        super(ResNet34Backbone, self).__init__()
        self.model = (
            models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            if pretrained
            else models.resnet34()
        )
        self.model.fc = nn.Identity()  # Remove the final fully connected layer

    def set_input_channels(self, channels: int):
        if channels == 1:
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

class ResNet18SSLBackbone(Backbone):
    def __init__(self, pretrained: bool = True):
        super(ResNet18SSLBackbone, self).__init__()
        self.model = timm.create_model("ssl_resnet18", pretrained=pretrained)
        self.model.fc = nn.Identity()
        self.model.eval()

    def set_input_channels(self, channels: int):
        if channels == 1:
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

class ResNet50DINOBackbone(Backbone):
    def __init__(self, pretrained: bool = True):
        super(ResNet50DINOBackbone, self).__init__()
        self.model = models.resnet50(weights=None)

        if pretrained:
            state_dict = _load_checkpoint_from_path_or_url("https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth")
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("student."):
                    key = key[len("student.") :]
                if key.startswith("teacher."):
                    key = key[len("teacher.") :]
                if key.startswith("fc."):
                    continue
                cleaned_state_dict[key] = value

            self.model.load_state_dict(cleaned_state_dict, strict=False)

        self.model.fc = nn.Identity()

    def set_input_channels(self, channels: int):
        if channels == 1:
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )


class CLIPBackbone(Backbone):
    DEFAULT_ARCH = "ViT-B-32"
    DEFAULT_PRETRAINED = "openai"

    def __init__(self, pretrained: bool = True):
        super(CLIPBackbone, self).__init__()
        self.backbone_type = "clip"

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "CLIP backbone requires open-clip-torch. Install with: pip install open-clip-torch"
            ) from exc

        arch = os.getenv("CLIP_ARCH", self.DEFAULT_ARCH).strip()
        pretrained_tag = os.getenv("CLIP_PRETRAINED", self.DEFAULT_PRETRAINED).strip()
        if not pretrained:
            pretrained_tag = None

        self.model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained_tag
        )
        self.model.eval()

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model.encode_image(x).float()

    def set_input_channels(self, channels: int):
        if channels == 1:
            raise ValueError("CLIP backbones require 3-channel RGB inputs.")


class CLIPViTB32Backbone(CLIPBackbone):
    DEFAULT_ARCH = "ViT-B-32"
    DEFAULT_PRETRAINED = "openai"


class CLIPResNet50Backbone(CLIPBackbone):
    DEFAULT_ARCH = "RN50"
    DEFAULT_PRETRAINED = "openai"
