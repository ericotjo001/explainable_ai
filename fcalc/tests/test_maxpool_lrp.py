import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
from models.networks_LRP import MaxPool2d_LRP


kernel_size = 2
mp = MaxPool2d_LRP(kernel_size, 
	stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

x = torch.empty(2,1,6,5).uniform_(0, 1)
y = mp(x)
RZ = F.avg_pool2d(y, kernel_size, stride=None, padding=0) 
print(x)
print(y)
print(RZ)

mp.forward_lrp(x)
R1 = mp.relprop1_debug(y, tab_level=0, verbose=250)
R2 = mp.relprop1(y)
print(R2.shape)