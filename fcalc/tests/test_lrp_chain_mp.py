import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
from models.networks_components2 import ConvChain2D_002mp
is_first_layer = False #  {'min':0.,'max':1.} # False
in_channel = 3
out_channel = in_channel
kernel_size = 3
x = torch.tensor(np.random.normal(0,1,size=(1,in_channel,28,28)),dtype=torch.float)

cc = ConvChain2D_002mp(in_channel, out_channel, kernel_size, chain_length=3,mp_kernel_size=2,
	stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
	mp_stride = None, mp_padding=0, mp_dilation=1,
	layer_name='convb', is_first_layer=is_first_layer)

cc.forward_debug(x, tab_level=0, verbose=250)
cc.forward_lrp(x)
y = cc(x)
print('y.shape:%s'%(str(y.shape)))

lrp_mode='relprop1'
R = cc.relprop_debug(y, mode=lrp_mode, tab_level=1, verbose=250)
print("No DEBUG:")
R = cc.relprop(y, mode=lrp_mode)
print(R.shape)