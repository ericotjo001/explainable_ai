
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
from utils.lrp_utils import partial_amplify


x = np.random.randint(-10,10,size=(3,2,10), )
x = torch.tensor(x)

print(x)
y = partial_amplify(x, alpha=0.5, amp=2., verbose=0)
print(y)