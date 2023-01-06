# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch
import torch.nn as nn
import torchvision.models as mod

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
        
