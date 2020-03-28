
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

x = torch.tensor(np.random.normal(0,1,size=(1,10,1,1)),dtype=torch.float)

x1 = x.clone().squeeze()
print(x1.shape)
x2 = x.clone().squeeze(3).squeeze(2)
print(x2.shape)
x3 = x.clone().squeeze(0)
print('x3.shape:',x3.shape)

x1 = torch.tensor(np.random.normal(0,1,size=(1,1,4,3)),dtype=torch.float)
print(x1.squeeze(0).shape)