import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
from models.networks_LRP import Conv2d_LRP
from models.networks_components import ConvBlock2D_001

tab_level=0

this_device = 'cuda:0' # 'cuda:0', None
x = torch.tensor(np.random.normal(0,1,size=(1,24,28,28)),dtype=torch.float).to(device=this_device)

pm.printvm("Conv", tab_level=tab_level)
cv = Conv2d_LRP(24, 24, 3, 
	stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros').to(device=this_device)
y = cv.forward_lrp(x.to(device=this_device))
pm.printvm('y.shape:%s'%(str(y.shape)), tab_level=1)
R = cv.relprop2_debug(y, tab_level=tab_level+1, verbose=250)

pm.printvm("ConvBlock", tab_level=tab_level)
cvb = ConvBlock2D_001(24, 24, 3, 
	stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros').to(device=this_device)
cvb.forward_lrp(x)
cvb.relprop_debug(R, mode='relprop2_debug', tab_level=tab_level+1, verbose=250)

pm.printvm("ConvBlock (not debug)", tab_level=tab_level)
cvb.forward_lrp(x)
cvb.relprop(R, mode='relprop2')

import utils.lrp_utils as lu
R = lu.force_fit_size_x_to_y_2D(R, x)
pm.printvm('R.shape:%s'%(str(R.shape)), tab_level=1)
