import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

n_channel = 2
x = torch.tensor(np.random.randint(0,10,size=(1,n_channel,6,6)),dtype=torch.float)

cv = nn.Conv2d(n_channel, 1, 3, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
temp = cv.weight*0-1
temp[0,:,1,1] = 1.0
temp = temp/n_channel
cv.weight = torch.nn.Parameter(data=temp,requires_grad=False)
# cv.reset_parameters()
print(cv.weight)
print(cv.bias)

y=cv(x)
print(x)
print(y)