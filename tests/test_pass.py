
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
from utils.lrp_utils import fraction_pass


x = np.random.randint(-10,10,size=(3,2,10), )
x = torch.tensor(x)

print(x)
y = fraction_pass(x, alpha1=0.,alpha2=0.5, verbose=0)
print(y)