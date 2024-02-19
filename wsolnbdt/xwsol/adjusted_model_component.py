# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch
import torch.nn as nn
import torchvision.models as mod

from typing import *


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlockAdjusted(mod.resnet.BasicBlock):
    def __init__(self):
        super(BasicBlockAdjusted, self).__init__(3, 3,) # dummy inputs
        # we will use inherit_weights() to get all its components
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

    def inherit_weights(self, BasicBlock):
        self.conv1 = BasicBlock.conv1
        self.bn1 = BasicBlock.bn1
        self.conv2 = BasicBlock.conv2
        self.bn2 = BasicBlock.bn2
        self.downsample = BasicBlock.downsample
        self.stride = BasicBlock.stride
        
class BottleneckAdjusted(nn.Module):

    expansion: int = 4

    def __init__(self,) -> None:
        super().__init__()

        self.conv1 = None # conv1x1(inplanes, width)
        self.bn1 = None # norm_layer(width)
        self.conv2 = None # conv3x3(width, width, stride, groups, dilation)
        self.bn2 = None # norm_layer(width)
        self.conv3 = None # conv1x1(width, planes * self.expansion)
        self.bn3 = None # norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.downsample = None # downsample
        self.stride = None #stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

    def inherit_weights(self, Bottleneck):
        self.conv1 = Bottleneck.conv1
        self.bn1 = Bottleneck.bn1
        self.conv2 = Bottleneck.conv2
        self.bn2 = Bottleneck.bn2
        self.conv3 = Bottleneck.conv3
        self.bn3 = Bottleneck.bn3
        self.downsample = Bottleneck.downsample
        self.stride = Bottleneck.stride