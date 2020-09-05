import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *


x = torch.tensor(np.random.normal(0,1,size=(5,5)),dtype=torch.float)
print(x)
print(x*(x>0).to(x.dtype))

