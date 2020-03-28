
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

BATCH_SIZE = 2
x = torch.tensor(np.random.normal(0,1,size=(BATCH_SIZE,4)),dtype=torch.float)
x[1,2]=999

print(x)
argma = torch.argmax(x.clone())
print(argma)
try:
	print(x[argma])
except:
	print('This gives error with BATCH_SIZE=2')