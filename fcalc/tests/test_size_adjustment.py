import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
import utils.lrp_utils as lu


this_device = None # 'cuda:0', None
x = torch.tensor(np.random.normal(0,1,size=(2,3,30,40)),dtype=torch.float).to(device=this_device)
y = torch.tensor(np.random.normal(0,1,size=(2,3,36,36)),dtype=torch.float).to(device=this_device)

x1,y1 = lu.relprop_size_adjustment_2D(x,y)
print(x1.shape)
print(y1.shape)