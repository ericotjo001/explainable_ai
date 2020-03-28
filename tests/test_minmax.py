import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

x = np.random.randint(-10,10,size=(4,5))
y = np.random.randint(-10,10,size=(4,5))
compare = (x>y).astype(np.float)
z = x*compare + y*(1-compare)
print(x)
print(y)
print(compare)
print(z)
print(z>=x)
print(z>=y)