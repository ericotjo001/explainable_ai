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

	LIST_OF_LAYERS_TO_OBSERVE = ['fn'+str(i) for i in range(1, 3+1)] + ['convb_'+str(i) for i in range(1,5+1)] 
	LIST_OF_DATA_NAME = ['positive_mean_power', 'negative_mean_power', 'average_mean_power']

	"""
	ORDERED_LIST_OF_GROUNDTRUTH_TO_OBSERVE
	  for cifar, if we set range(10), we observe from 0,1,...,9 (collected data may be unordered)
	"""
	ORDERED_LIST_OF_GROUNDTRUTH_TO_OBSERVE = range(10) 
	# LIST_OF_LAYERS_TO_OBSERVE_2 =  ['convb_3'] # 'fn2' #
	LIST_OF_LAYERS_TO_OBSERVE_2 = ['fn'+str(i) for i in range(1, 3+1)] + ['convb_'+str(i) for i in range(1,5+1)] # ['fn1'] # 
	LIST_OF_DATA_NAME_2 = [
		'mean_of_highest_mean_power', # float
		'mean_of_lowest_mean_power', # float
		'mean_of_R_highest_mean_power', # np.array
		'mean_of_R_lowest_mean_power', # np.array
		'sd_of_R_highest_mean_power', # np.array		
		'sd_of_R_lowest_mean_power', # np.array	

		# 'mean_sqr_of_R_highest_mean_power', #np.array
		# 'mean_sqr_of_R_lowest_mean_power', #np.array
	]

	tracker_name = 'drivethru_0001_alexnet_cifar'
	drivethru_visual_abstract_0001(config_data, tracker_name, LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME,
		ORDERED_LIST_OF_GROUNDTRUTH_TO_OBSERVE, LIST_OF_LAYERS_TO_OBSERVE_2, LIST_OF_DATA_NAME_2,
		DEBUG_FAST_FORWARD,
		tab_level=0, verbose=250)