import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

x = np.random.normal(0,1,size=(3,4))
x[0,0] = np.inf
x[1,1] = -np.inf
x = torch.tensor(x,dtype=torch.float)
print(x)


print(torch.max(x))
print(torch.min(x))