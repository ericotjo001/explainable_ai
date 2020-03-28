import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
from utils.metrics_for_lrp_output import *

pmp = PercentileMeanPower()
x = torch.tensor(np.random.normal(0,1,size=(3,28,28)),dtype=torch.float)

SWITCH = 0
if SWITCH ==0:
	positive_mean_power, negative_mean_power = pmp.sorted_sign_split(x, mp_threshold=0.8, verbose=250, tab_level=0)
	print(positive_mean_power, negative_mean_power)
	
elif SWITCH==1:
	x = np.random.randint(-10,10,size=(20))

	print(x)
	print(x[10:])
	print(x[:10])
	# [-4  1 -1  7  2  7 -1 -5  0 -3  1 -9  6 -4 -3  4 -2  8  1  5]
	# [ 1 -9  6 -4 -3  4 -2  8  1  5]
	# [-4  1 -1  7  2  7 -1 -5  0 -3]
elif SWITCH==2:
	x = np.random.randint(0,10,size=(1))
	x = np.sort(x)
	print(x)
	pmp.compute_mean_power(x, threshold=0.7, verbose=250, tab_level=1)
elif SWITCH==3:
	pmp.compute_mean_power_example(sample_size=100, array_size=28*28, threshold=0.8)
	plt.show()

