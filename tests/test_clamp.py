import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

x = np.random.randint(0,10,size=(3,6), )
x[0,1] = 10000
x[1,1] = -10000
x = torch.tensor(x)
print(x)
print(torch.clamp(x,-777,777))