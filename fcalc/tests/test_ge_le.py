import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

x = np.random.normal(0,1,size=(2,5))
x[0,2] = np.nan
print(x)

x>1
x<=0

print(x[~np.isnan(x)])