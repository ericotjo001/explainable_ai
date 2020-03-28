import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

import utils.lrp_utils as lu

x = torch.tensor(np.random.normal(0,1,size=(3,4)),dtype=torch.float)
print(x)
small_number=0.5
signed_small_x = lu.find_signed_small_x(x, small_number)
print(signed_small_x)

print(x + 1000*signed_small_x)