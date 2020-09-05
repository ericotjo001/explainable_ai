from utils.utils import *
from utils.logger import TimePrinter
from utils.logger import Logger
from pipeline.custom_sequence.sequence2 import print_time_collect
import pipeline.drivethru.drivethru_smallnet_mnist as smut
import pipeline.custom_sequence.sequence_utils as ut

import visual.drivethru_visual_smallnet_mnist_preprocessing as vdp
import visual.drivethru_visual_smallnet_mnist_functions as vd
import visual.drivethru_visual_smallnet_mnist_functions3 as vd3

tp = TimePrinter()

def seq_smallnet_mnist_upsize(config_data, tab_level=0, verbose=250):
	print('seq_smallnet_mnist_upsize()')
	config_data['data_from_torch']['mnist']['resize'] = (140,140)

	config_data['NO_OF_RUNS'] = 1
	config_data['general']['epoch'] = 1  # max 2 per 24 hours
	config_data['drivethru']['eval_every_n_iter'] = 1000
	config_data['drivethru']['n_of_test_data_per_eval'] = 1000
	config_data['drivethru']['n_of_test_data_per_LRP_eval'] = 240
	seq_smallnet_mnist_general(config_data, tab_level=0, verbose=250)

def seq_smallnet_mnist(config_data, tab_level=0, verbose=250):
	print('seq_smallnet_mnist()')
	config_data['data_from_torch']['mnist']['resize'] = None

	config_data['NO_OF_RUNS'] = 1
	config_data['general']['epoch'] = 1  # NO max per 24 hours
	config_data['drivethru']['eval_every_n_iter'] = 1000
	config_data['drivethru']['n_of_test_data_per_eval'] = 1000
	config_data['drivethru']['n_of_test_data_per_LRP_eval'] = 240
	seq_smallnet_mnist_general(config_data, tab_level=0, verbose=250)

def seq_smallnet_mnist_general(config_data, tab_level=0, verbose=250):
	# model/data specific
	tracker_name = 'drivethru_0001_smallnet_mnist'
	LIST_OF_LAYERS_TO_OBSERVE = ['fn'+str(i) for i in range(1, 3+1)] + ['convb_'+str(i) for i in range(1,6+1)] 
	LIST_OF_DATA_NAME = ['positive_mean_power', 'negative_mean_power', 'average_mean_power']
	import pipeline.drivethru.drivethru_smallnet_mnist as sm
	this_drivethru_method = sm.drivethru_0001_smallnet_mnist
	
	if config_data['console_subsubmode'] is None:
		print('Set --subsubmode label_name. For example --subsubmode RXXXX1')
		return
	config_data['training']['series_name'] = config_data['console_subsubmode']
	config_data['visual']['series_name'] = config_data['console_subsubmode']
	
	if IS_DEBUG:
		config_data['NO_OF_RUNS'] = 1
		config_data['drivethru']['eval_every_n_iter'] = 7
		config_data['drivethru']['n_of_test_data_per_eval'] = 100
		config_data['drivethru']['n_of_test_data_per_LRP_eval'] = 24
	
	time_collect = {'drivethru': {},'visual':0}
	time_collect = running_drivethru(config_data, this_drivethru_method, time_collect, tracker_name,
		verbose=verbose, tab_level=tab_level)
	time_collect = running_visuals(config_data,LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME, 
		tracker_name, time_collect,	verbose=verbose, tab_level=tab_level)

	print_time_collect(time_collect)

def running_drivethru(config_data, this_drivethru_method, time_collect, tracker_name,
	verbose=0, tab_level=0):	
	state_tracker = smut.setup_state_tracker(config_data, tracker_name, for_training=True, verbose=verbose, tab_level=tab_level)
	full_path_log_file = ut.get_model_folder_path(config_data, state_tracker, tab_level=tab_level+1)
	sys.stdout = Logger(full_path_log_file=full_path_log_file)

	NO_OF_RUNS = config_data['NO_OF_RUNS']
	for i in range(1, 1 + NO_OF_RUNS):
		start = time.time()

		this_drivethru_method(config_data, tab_level=0, verbose=250)

		end = time.time()
		time_secs = end - start
		tp.print_smh('run_time',start, end, if_x_times=(10.,100.),
			verbose=verbose, tab_level=tab_level, verbose_threshold=None)
		time_collect['drivethru'][i] = time_secs
		ut.print_partition()
	return time_collect


def running_visuals(config_data, LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME, 
	tracker_name, time_collect, verbose=0, tab_level=0):
	t_vis1 = time.time()
	arranged_data, state_tracker = vdp.arrange_by_iter(tracker_name, config_data, 
		DEBUG_OVERVIEW_ONLY=False, verbose=250, tab_level=tab_level)
	path_to_save_folder = vd.get_path_to_save_folder(state_tracker)
	vd.plot_loss(arranged_data['loss_data_by_epoch'], path_to_save_folder)
	vd.plot_accuracy(arranged_data['accuracy_vs_iter'], path_to_save_folder)
	vd.visualize_mean_power_by_layer_over_iter(arranged_data['processed_datapoint_vs_iter'], 
		path_to_save_folder, LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME, verbose=100, tab_level=0)
	vd3.visualize_filter_effect_over_iter(arranged_data['saliency_by_groundtruth_vs_iter'],
		path_to_save_folder, verbose=250, tab_level=0)	
	t_vis2 = time.time()
	time_collect['visual'] = t_vis2 - t_vis1 # seconds
	return time_collect