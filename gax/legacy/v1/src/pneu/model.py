import numpy as np

import torch
import torch.nn as nn
import torchvision.models as mod

from .adjusted_model_component import BasicBlockAdjusted
class Resnet34Pneu(nn.Module):
    def __init__(self):
        super(Resnet34Pneu, self).__init__()
        self.iter = nn.Parameter(torch.from_numpy(np.array([0.])), requires_grad=False)
    
        self.backbone = mod.resnet34(pretrained=True, progress=False)
        self.fc = nn.Linear(512, 2, bias=False)

        self.adjust_for_captum_problem()
        # self.print_after_captum_adjustment() # or debugging

    def forward(self,x):
        for i,m in enumerate(self.backbone.children()):
            if i>8: break
            x = m(x)
        # print(x.shape) # torch.Size([4, 512, 1,1])
        x = x.squeeze(3).squeeze(2)
        x = self.fc(x)
        # print(x.shape) # torch.Size([4, 2])
        return x

    def adjust_for_captum_problem(self):
        setattr(self.backbone,'relu',nn.ReLU() )
        for i,(layer_name,m) in enumerate(self.backbone.named_children()):    
            if type(m) == nn.Sequential:
                for j,(sublayer_name,m2) in enumerate(m.named_children()):
                    if type(m2) == mod.resnet.BasicBlock:
                        # setattr(getattr(getattr(self.backbone, layer_name), sublayer_name),'relu', nn.ReLU())
                        temp = BasicBlockAdjusted()
                        temp.inherit_weights(m2)
                        setattr(getattr(self.backbone, layer_name), sublayer_name, temp)

    def print_after_captum_adjustment(self):
        for i,(layer_name,m) in enumerate(self.backbone.named_children()):
            print(type(m))
            for j,(sublayer_name,m2) in enumerate(m.named_children()):
                print(type(m2))
                for k,(subsublayer_name,m3) in enumerate(m2.named_children()):
                    print(type(m3))

class AlexPneu(nn.Module):
    def __init__(self,):
        super(AlexPneu, self).__init__()
        self.iter = nn.Parameter(torch.from_numpy(np.array([0.])), requires_grad=False)
    
        self.backbone = mod.alexnet(pretrained=True, progress=False)
        self.fc= nn.Linear(4096,2,bias=False)    
        self.adjust_for_captum()

    def forward(self,x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        for i,m in enumerate(self.backbone.classifier.children()):
            if i==6: break # skip last layer with 1000 classes, change it with your own linear layer
            x = m(x)
        x = self.fc(x)
        return x   

    def adjust_for_captum(self):
        # deal with all the inplace relu problems
        for i,m in enumerate(self.backbone.features.children()):
            if type(m)==nn.ReLU:
                # print('yes', type(self.backbone.features[i]))
                self.backbone.features[i] = nn.ReLU() # without inplace=True
        for i,m in enumerate(self.backbone.classifier.children()):
            if type(m)==nn.ReLU:
                self.backbone.classifier[i] = nn.ReLU() # without inplace=True

class Generator(nn.Module):
    def __init__(self, img_size, ):
        super(Generator, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(3,)+img_size)+ 1)
        self.b = nn.Parameter(torch.zeros(size=(3,)+img_size)+ 0.01)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.W * x + self.b
        x = self.act(x)
        return x

