import torch.nn as nn

# the original BasicBlock is from the following
# https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html

class BasicBlockRevised(nn.Module):
    expansion = 1

    def __init__(self, BasicBlock):
        super(BasicBlockRevised, self).__init__()

        self.conv1 = BasicBlock.conv1
        self.bn1 = BasicBlock.bn1
        self.relu = BasicBlock.relu
        self.conv2 = BasicBlock.conv2
        self.bn2 = BasicBlock.bn2
        self.downsample = BasicBlock.downsample
        self.stride = BasicBlock.stride

        # we make sure different relu has different implementation
        # this is so that DeepLift can assign the correct input/output at the correct places
        self.relu2 = nn.ReLU(inplace=True) # revised
        # these are the input/output attributes required by DeepLift
        self.relu.input = None
        self.relu.output = None
        self.relu2.input = None
        self.relu2.output = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out) # revised

        return out