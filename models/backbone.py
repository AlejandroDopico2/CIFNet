from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor
from torchvision import models


class Backbone(ABC, nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def set_input_channels(self, channels: int):
        pass


class ResNetBackbone(Backbone):
    def __init__(self, pretrained: bool = True):
        super(ResNetBackbone, self).__init__()
        self.model = (
            models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            if pretrained
            else models.resnet18(weights=None)
        )
        self.model.fc = nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        return self.model(x)

    def set_input_channels(self, channels: int):
        if channels == 1:
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )


class MobileNetBackbone(Backbone):
    def __init__(self, pretrained: bool = True):
        super(MobileNetBackbone, self).__init__()
        self.model = (
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            if pretrained
            else models.mobilenet_v2(weights=None)
        )
        self.model.classifier = nn.Identity()  # Remove the final classifier layer

    def forward(self, x):
        return self.model(x)

    def set_input_channels(self, channels: int):
        self.model.features[0][0] = nn.Conv2d(
            channels,
            self.model.features[0][0].out_channels,
            kernel_size=self.model.features[0][0].kernel_size,
            stride=self.model.features[0][0].stride,
            padding=self.model.features[0][0].padding,
            bias=self.model.features[0][0].bias,
        )

class CustomBackbone(Backbone):
    def __init__(self, pretrained):
        super(CustomBackbone, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def set_input_channels(self, channels: int):
        self.features[0]= nn.Conv2d(
            channels,
            self.features[0].out_channels,
            kernel_size=self.features[0].kernel_size,
            stride=self.features[0].stride,
            padding=self.features[0].padding,
        )

    def forward(self, x):
        return self.features(x)