from utils.utils import *
from utils.result_wrapper import ResultWrapper

from visual.drivethru_visual_smallnet_mnist_functions import *
# from visual.drivethru_visual_smallnet_mnist_functions2 import *
from visual.drivethru_visual_smallnet_mnist_functions3 import *
from visual.drivethru_visual_smallnet_mnist_preprocessing import *

if IS_DEBUG:
	DEBUG_OVERVIEW_ONLY = 0
	DEBUG_FAST_FORWARD = [1,1,1,0]  # if skip, set to 1
else:
	DEBUG_OVERVIEW_ONLY = False # (bool)
	DEBUG_FAST_FORWARD = [False, False, False, False] # list of bools

def drivethru_visual0001(config_data, tab_level=0, verbose=250):
	print('drivethru_visual0001(). config_data[visual][series_name]:%s'%(str(config_data['visual']['series_name'])))
	print('\n** MANY HARDCODED VARIABLES\n')

	LIST_OF_LAYERS_TO_OBSERVE = ['fn'+str(i) for i in range(1, 3+1)] + ['convb_'+str(i) for i in range(1,6+1)] 
	LIST_OF_DATA_NAME = ['positive_mean_power', 'negative_mean_power', 'average_mean_power']

	tracker_name = 'drivethru_0001_smallnet_mnist'
	drivethru_visual_abstract_0001(config_data, tracker_name, LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME,
		DEBUG_FAST_FORWARD,
		tab_level=0, verbose=250)

def drivethru_visual_abstract_0001(config_data, tracker_name, LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME,
	DEBUG_FAST_FORWARD,
	tab_level=0, verbose=250):
	arranged_data, state_tracker = arrange_by_iter(tracker_name, config_data, 
		DEBUG_OVERVIEW_ONLY=DEBUG_OVERVIEW_ONLY, verbose=250, tab_level=tab_level)
	path_to_save_folder = get_path_to_save_folder(state_tracker)

	if not DEBUG_FAST_FORWARD[0]:
		plot_loss(arranged_data['loss_data_by_epoch'], path_to_save_folder)
	if not DEBUG_FAST_FORWARD[1]:
		plot_accuracy(arranged_data['accuracy_vs_iter'], path_to_save_folder)
	if not DEBUG_FAST_FORWARD[2]:
		visualize_mean_power_by_layer_over_iter(arranged_data['processed_datapoint_vs_iter'], 
			path_to_save_folder, LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME, verbose=100, tab_level=0)
	if not DEBUG_FAST_FORWARD[3]:
		visualize_filter_effect_over_iter(arranged_data['saliency_by_groundtruth_vs_iter'],
			path_to_save_folder, verbose=250, tab_level=0)
