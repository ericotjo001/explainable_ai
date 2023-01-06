import numpy as np

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_size, ):
        super(Generator, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(3,)+img_size)+ 1)
        self.act = nn.Tanh()


    def forward(self, x):
        x = self.W* x
        x = self.act(x)
        return x


