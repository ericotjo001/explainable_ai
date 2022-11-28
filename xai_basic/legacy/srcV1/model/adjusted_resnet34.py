import torch
import torch.nn as nn
import torchvision.models as mod
from .torchvision_resnet_components_revised import BasicBlockRevised

class AdjResnet34(nn.Module):
    def __init__(self):
        super(AdjResnet34, self).__init__()

        self.backbone = mod.resnet34(pretrained=True)
        self.fc = nn.Conv2d(512,10,1)

    def readjust(self, verbose=None):
        # this is readjustment for DeepLift's complain that ReLU modules do NOT have input/output attributes
        for i,m in enumerate(self.backbone.children()):
            if isinstance(m, nn.ReLU):
                # print('[%s] is relu'%(str(i)))
                setattr(m,'input',None)
                setattr(m,'output',None)
            number_of_subchildren = len(list(m.children()))
            if number_of_subchildren>0:
                for j, (name,m2) in enumerate(m.named_children()):
                    # print(name, type(m2), type(getattr(m,name)))
                    if isinstance(getattr(m, name), mod.resnet.BasicBlock):
                        setattr(m,name, self.replace_basic_block_for_captum_compatibility(m2)) 
                        # print(m2.relu)
        if verbose is not None:
            if verbose>=100:
                self.readjust_print_debug()

    def forward(self, x):
        for i, m in enumerate(self.backbone.children()):
            if i==9: break # x = x.view(1,-1)
            x = m(x)
            # print(i, x.shape)
        x = self.fc(x)
        x = x.squeeze(3).squeeze(2)
        return x

    ##################################################
    # fixes for DeepLift in Captum
    ##################################################
    
    def replace_basic_block_for_captum_compatibility(self, basicblock):
        new_basic_block = BasicBlockRevised(basicblock)
        return new_basic_block

    def readjust_print_debug(self):
        print('\nDOUBLE CHECK')
        for i, m in enumerate(self.backbone.children()):
            print(type(m))
            number_of_subchildren = len(list(m.children()))
            if number_of_subchildren>0:
                for j,m2 in enumerate(m.children()):
                    print('  ',type(m2))

        print('named_children names:')
        for i, (name, m) in enumerate(self.backbone.named_children()):
            print(name)

    def select_first_layer(self):
        return self.backbone.conv1