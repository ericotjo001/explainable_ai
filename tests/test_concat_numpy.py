import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

x = []
for i in range(3):
	x.append(np.random.normal(0,1,size=(3,5,5)))

for y in x:
	print(y.shape)

x1 = np.concatenate(x,axis=0)
print('x1.shape:',x1.shape)

x1b = np.concatenate(x,axis=1)
print('x1b.shape:',x1b.shape)

x2 = np.stack(x,axis=0)
print('x2.shape', x2.shape)