from utils.utils import *
from pipeline.custom_sequence.sequence2 import * # custom_sequence_abstract_0002

if IS_DEBUG:
	AUTO_MODE = False
else:
	AUTO_MODE = True


def custom_sequence_0003b(config_data, tab_level=0, verbose=250):
	print('custom_sequence_0003b(). CIFAR. Alexnet')

	tracker_name = 'drivethru_0001_alexnet_cifar'

	LIST_OF_LAYERS_TO_OBSERVE = ['fn'+str(i) for i in range(1, 3+1)] + ['convb_'+str(i) for i in range(1,5+1)] 
	LIST_OF_DATA_NAME = ['positive_mean_power', 'negative_mean_power', 'average_mean_power']

	if AUTO_MODE:
		custom_sequence_abstract_0002(config_data, tracker_name, 
			LIST_OF_LAYERS_TO_OBSERVE,LIST_OF_DATA_NAME,
			tab_level=tab_level, verbose=verbose)
	else:
		# customize your mode here
		import pipeline.drivethru.drivethru_smallnet_mnist as smut
		import pipeline.drivethru.drivethru_alexnet_cifar as sm

		series_code = config_data['console_subsubmode']
		config_data['training']['series_name'] = 'Custom_' + str(series_code)
		config_data['visual']['series_name'] = 'Custom_' + str(series_code)

		state_tracker = smut.setup_state_tracker(config_data, tracker_name, for_training=True, verbose=verbose, tab_level=tab_level)
		# sm.drivethru_0001_smallnet_cifar(config_data, tab_level=0, verbose=250)

		arranged_data, state_tracker = arrange_by_iter(tracker_name, config_data, 
			DEBUG_OVERVIEW_ONLY=False, verbose=250, tab_level=tab_level)
		path_to_save_folder = get_path_to_save_folder(state_tracker)
		# plot_loss(arranged_data['loss_data_by_epoch'], path_to_save_folder)
		# plot_accuracy(arranged_data['accuracy_vs_iter'], path_to_save_folder)
		# visualize_mean_power_by_layer_over_iter(arranged_data['processed_datapoint_vs_iter'], 
		# 	path_to_save_folder, LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME, verbose=100, tab_level=0)
		visualize_filter_effect_over_iter(arranged_data['saliency_by_groundtruth_vs_iter'],
			path_to_save_folder, verbose=250, tab_level=0)