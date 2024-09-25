from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor
from torchvision import models


class Backbone(ABC, nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

    @abstractmethod
    def set_input_channels(self, channels: int):
        pass

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ResNetBackbone(Backbone):
    def __init__(self, pretrained: bool = True):
        super(ResNetBackbone, self).__init__()
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


class MobileNetBackbone(Backbone):
    def __init__(self, pretrained: bool = True):
        super(MobileNetBackbone, self).__init__()
        self.model = (
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            if pretrained
            else models.mobilenet_v2()
        )
        self.model.classifier = nn.Identity()  # Remove the final classifier layer

    def set_input_channels(self, channels: int):
        self.model.features[0][0] = nn.Conv2d(
            channels,
            self.model.features[0][0].out_channels,
            kernel_size=self.model.features[0][0].kernel_size,
            stride=self.model.features[0][0].stride,
            padding=self.model.features[0][0].padding,
            bias=self.model.features[0][0].bias,
        )


class DenseNetBackbone(Backbone):
    def __init__(self, pretrained: bool = True):
        super(DenseNetBackbone, self).__init__()
        self.model = (
            models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            if pretrained
            else models.densenet121()
        )
        self.model.classifier = nn.Identity()  # Remove the final classifier layer

    def set_input_channels(self, channels: int):
        self.model.features[0] = nn.Conv2d(
            channels,
            self.model.features[0].out_channels,
            kernel_size=self.model.features[0].kernel_size,
            stride=self.model.features[0].stride,
            padding=self.model.features[0].padding,
            bias=self.model.features[0].bias,
        )


class CustomBackbone(Backbone):
    def __init__(self, pretrained):
        super(CustomBackbone, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def set_input_channels(self, channels: int):
        self.model[0] = nn.Conv2d(
            channels,
            self.model[0].out_channels,
            kernel_size=self.model[0].kernel_size,
            stride=self.model[0].stride,
            padding=self.model[0].padding,
        )
