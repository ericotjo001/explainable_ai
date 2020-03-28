import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

x = torch.tensor(range(10),dtype=torch.float,requires_grad=True)
z = x.data
y1 = torch.sum(z**2)
y = torch.sum(x**2)
y.backward()

print(x)
print('x.grad:',x.grad)



x1 = torch.tensor(range(10),dtype=torch.float,requires_grad=True)
print(x1)
x2 = x1.clone().detach()
x3 = x1.clone().detach()
x1[4] = 1000
x3[1]=50
print('='*80)
print(x1)
print(x2)
print(x3)