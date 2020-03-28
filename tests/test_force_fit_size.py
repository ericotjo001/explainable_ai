import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
import utils.lrp_utils as lu


this_device = None
x = torch.tensor(np.random.randint(0,10,size=(1,3,8,9))).to(device=this_device)
y = torch.tensor(np.random.randint(0,10,size=(1,3,6,6))).to(device=this_device)


x1 = lu.force_fit_size_x_to_y_2D(x, y)

print(x)
print(x1)