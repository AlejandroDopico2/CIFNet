
from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor
from torchvision import models
import torch
import timm

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
        # self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
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
        self.model.fc = nn.Identity()  # Remove the final fully connected layer

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
