import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
from models.networks_components2 import ConvBlock2D_002mp

is_first_layer =  {'min':0.,'max':1.}
x = torch.tensor(np.random.normal(0,1,size=(1,3,28,28)),dtype=torch.float)
cv = ConvBlock2D_002mp(3, 24, 3, mp_kernel_size=2,
	stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros',
	mp_stride = None, mp_padding=0, mp_dilation=1,
	is_first_layer=is_first_layer)

y = cv.forward_debug(x)
print("initiate self.x")
cv.forward_lrp(x)
print("  done")
R = cv.relprop_debug(y, mode='relprop2', tab_level=0, verbose=250)
R1 = cv.relprop(y, mode='relprop2')
print('R1.shape:',R1.shape)