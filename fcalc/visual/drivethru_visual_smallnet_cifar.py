from utils.utils import *
from utils.result_wrapper import ResultWrapper

from visual.drivethru_visual_smallnet_mnist import drivethru_visual_abstract_0001

if IS_DEBUG:
	DEBUG_OVERVIEW_ONLY = 0
	DEBUG_FAST_FORWARD = [1,1,1,0] # if skip, set to 1
else:
	DEBUG_OVERVIEW_ONLY = False # (bool)
	DEBUG_FAST_FORWARD = [False, False, False, False] # list of bools

def drivethru_visual0001(config_data, tab_level=0, verbose=250):
	print('drivethru_visual0001(). config_data[visual][series_name]:%s'%(str(config_data['visual']['series_name'])))
	print('\n** MANY HARDCODED VARIABLES\n')

	LIST_OF_LAYERS_TO_OBSERVE = ['fn'+str(i) for i in range(1, 3+1)] + ['convb_'+str(i) for i in range(1,6+1)] 
	LIST_OF_DATA_NAME = ['positive_mean_power', 'negative_mean_power', 'average_mean_power']

	tracker_name = 'drivethru_0001_smallnet_cifar'
	drivethru_visual_abstract_0001(config_data, tracker_name, LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME,
		DEBUG_FAST_FORWARD,
		tab_level=0, verbose=250)