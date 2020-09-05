import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
from models.networks_smallCNN_probe import SmallCNNprobe0001

x = torch.tensor(np.random.normal(0,1,size=(1,1,28,28)),dtype=torch.float)
cnn = SmallCNNprobe0001(IMG_SIZE=(28,28), 
	INPUT_CHANNEL_SIZE = 1,
	relprop_mode='relprop1', 
	set_default_params=True, 
	verbose=250, 
	tab_level=0).cpu()

cnn.eval()

x1 = cnn.convb_1(x)
x2 = cnn.convb_2(x1)
x3 = cnn.convb_3(x2)
x4 = cnn.convb_4(x3)
x5 = cnn.convb_5(x4)
x6 = cnn.convb_6(x5)

f1 = cnn.fn1(x6)
f2 = cnn.fn2(f1)
f3 = cnn.fn3(f2)
y = cnn(x)

check = np.all(f3.data.detach().numpy()==y.data.detach().numpy())
print(check)

cnn.forward_lrp(x)
R1 = cnn.relprop(y.data)
R2 = cnn.relprop(y.data)

relprop_mode = cnn.relprop_mode
R = cnn.fn3.relprop(y.data, mode=relprop_mode)
R = cnn.fn2.relprop(R, mode=relprop_mode)
R = cnn.fn1.relprop(R, mode=relprop_mode)
R = cnn.convb_6.relprop(R, mode=relprop_mode)
R = cnn.convb_5.relprop(R, mode=relprop_mode)
R = cnn.convb_4.relprop(R, mode=relprop_mode)
R = cnn.convb_3.relprop(R, mode=relprop_mode)
R = cnn.convb_2.relprop(R, mode=relprop_mode)
R = cnn.convb_1.relprop(R, mode=relprop_mode)

check2 = np.all(R1.detach().numpy()==R.detach().numpy())
check2b = np.all(R1.detach().numpy()==R2.detach().numpy())
print(check2, check2b)
dsqr = sum((R1.reshape(-1)-R.reshape(-1))**2)**0.5
print(dsqr/(28**2))
