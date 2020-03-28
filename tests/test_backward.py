import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

x = torch.tensor(np.random.randint(0,10, size=(2,3,2)), requires_grad=True, dtype=torch.float)
y = torch.sum(x*x)
print(x)
print(y)
y.backward()
print(x.grad)


print(x.sum())