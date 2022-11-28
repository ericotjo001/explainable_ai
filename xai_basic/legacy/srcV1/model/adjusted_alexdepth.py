from model.adjusted_alexnet import AdjAlexnet
import torch
import torch.nn as nn
import torchvision.models as mod

class AdjAlexDepth(AdjAlexnet):
    def __init__(self, config_data):
        super(AdjAlexDepth, self).__init__()

        if config_data is None:
            config_data = {
                'block_setting':    [
                    { 'conv_pos_args': (3,3,3), 'conv_kwargs':{'padding':2, 'dilation':2}},
                    { 'conv_pos_args': (3,3,3), 'conv_kwargs':{'padding':2, 'dilation':2}},
                ]
            }

        for i, block_setting in enumerate(config_data['block_setting']):
            cb = ConvBlock(*block_setting['conv_pos_args'], **block_setting['conv_kwargs'])
            setattr(self ,'block%s'%(str(i)), cb)

        self.n_blocks = len(config_data['block_setting'])

    def forward(self, x):
        # print(x.shape)
        for i in range(self.n_blocks):
            x = getattr(self, 'block%s'%(str(i)))(x)
            # print('[%s]'%(str(i)),x.shape)

        # raise Exception('DEBUGG')
        x = super(AdjAlexDepth, self).forward(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self,*args, **kwargs):
        super(ConvBlock, self).__init__()
        
        self.cv = nn.Conv2d( *args, **kwargs)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.cv(x)
        x = self.act(x)
        return x


