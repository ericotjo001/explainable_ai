import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
from utils.lrp_utils import fraction_clamp

x = np.random.randint(-10,10,size=(3,2,5), )
x = torch.tensor(x)

print(x)
y = fraction_clamp(x, alpha1=0.5,alpha2=1, verbose=0)
print(y)