import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

x = np.random.randint(0,10,size=(3,6)).astype(float)
y = np.random.randint(0,10,size=(3,6)).astype(float)
x[0,1] = - np.inf # np.nan
y[0,1] = np.nan

print(np.isnan(x))
print(np.any(np.isnan(x)))
print(np.isinf(x))
print(np.any(np.isinf(x)))

print("="*80)
print('is finite')
print(np.all(np.isfinite(x)))
print(np.all(np.isfinite(y)))


print("="*80)
x = torch.tensor(x)
print(x)
print(torch.sum(torch.isnan(x))>0)