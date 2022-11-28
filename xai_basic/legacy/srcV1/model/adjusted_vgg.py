import torch
import torch.nn as nn
import torchvision.models as mod

LITE_MODE = 0  # in the likely case that your local pc has not enough memory, set to 1 for testing
DEBUG_MODE = 0
if DEBUG_MODE: # freely toggle
    DEBUG_FOR_SMALL_RAM = 0
else: # DO NOT CHANGE
    DEBUG_FOR_SMALL_RAM = 0 # bool

class AdjVGG(nn.Module):
    def __init__(self):
        super(AdjVGG, self).__init__()
        self.backbone = mod.vgg16(pretrained=True)

        # self.cv = nn.Conv2d(512,64,(3,3))
        self.fc1 = nn.Linear(25088,4096) 
        self.act1 = nn.LeakyReLU()
        self.d1 = nn.Dropout()

        self.fc2 = nn.Linear(4096,10)
        self.act2 = nn.LeakyReLU()
        self.d2 = nn.Dropout()

        if LITE_MODE:
            print('    <<lite mode adjvgg used!>>')
            self.fc1 = nn.Linear(25088,1024) 
            self.fc2 = nn.Linear(1024,10)
        
        if DEBUG_FOR_SMALL_RAM:
            self.fc1 = nn.Linear(25088,128) 
            self.fc2 = nn.Linear(128,10)
        

    def forward(self, x):
        for i,m in enumerate(self.backbone.children()): 
            if i<2: # at i==2, the module is sequential FC for 1000 classes. Change this.
                x = m(x)

        # x = self.cv(x)
        s = x.shape # (batch, C, H, W)
        # print('A',x.shape)
        x = x.reshape(s[0],-1)
        # print('B',x.shape)
        # raise Exception('DEBUGGG')

        for i in ['1','2']:
            x = getattr(self,'d%s'%(str(i)))(x)
            x = getattr(self,'fc%s'%(str(i)))(x)
            x = getattr(self,'act%s'%(str(i)))(x)
        return x

    def select_first_layer(self):
        return getattr(self.backbone.features,'0')

# gg = AdjVGG()
# print(type(gg.select_first_layer()))