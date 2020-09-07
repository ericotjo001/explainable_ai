import torch
import torch.nn as nn
import torchvision.models as mod

class AdjAlexnet(nn.Module):
    def __init__(self):
        super(AdjAlexnet, self).__init__()
        self.backbone = mod.alexnet(pretrained=True,)

        self.d1 = nn.Dropout()
        self.fc1 = nn.Linear(9216,4096) 
        self.act1 = nn.LeakyReLU()

        self.d2 = nn.Dropout()
        self.fc2 = nn.Linear(4096,4096)
        self.act2 = nn.LeakyReLU()
        
        self.fc3 = nn.Linear(4096,10)

    def readjust(self, ):
        pass

    def forward(self, x):
        for i,m in enumerate(self.backbone.children()): 
            if i<2: # at i==2, the module is sequential FC for 1000 classes. Change this.
                x = m(x)

        s = x.shape # (batch, C, H, W)
        x = x.reshape(s[0],-1)

        for i in ['1','2']:
            x = getattr(self,'d%s'%(str(i)))(x)
            x = getattr(self,'fc%s'%(str(i)))(x)
            x = getattr(self,'act%s'%(str(i)))(x)
        x = self.fc3(x)
        return x

    def select_first_layer(self):
        return getattr(self.backbone.features,'0')


# gg = AdjAlexnet()
# print(type(gg.select_first_layer()))