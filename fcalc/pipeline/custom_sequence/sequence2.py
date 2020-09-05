from utils.utils import *
import pipeline.custom_sequence.sequence_utils as ut
from utils.logger import TimePrinter
tp = TimePrinter()

from visual.drivethru_visual_smallnet_mnist_functions import *
from visual.drivethru_visual_smallnet_mnist_functions2 import *
from visual.drivethru_visual_smallnet_mnist_functions3 import *
from visual.drivethru_visual_smallnet_mnist_preprocessing import *

def print_time_collect(time_collect):
	pm.printvm('print_time_collect()',tab_level=0, verbose=2, verbose_threshold=1)
	pm.printvm('drivethru times:',tab_level=1, verbose=2, verbose_threshold=1)	
	for run_number, time_secs in time_collect['drivethru'].items():
		tp.print_smh('drivethru_'+ str(run_number), 0, time_secs, if_x_times=(10.,100.),
			verbose=0, tab_level=2, verbose_threshold=None)		

	pm.printvm('Visualization time:',tab_level=1, verbose=2, verbose_threshold=1)
	tp.print_smh('vis_', 0, time_collect['visual'], if_x_times=(10.,100.),
		verbose=0, tab_level=2, verbose_threshold=None)	


# BELOW READY TO DEPRECATE


def custom_sequence_0002(config_data, tab_level=0, verbose=250):
	print('custom_sequence_0002(). MNIST. Smallnet')
	tracker_name = 'drivethru_0001_smallnet_mnist'

	LIST_OF_LAYERS_TO_OBSERVE = ['fn'+str(i) for i in range(1, 3+1)] + ['convb_'+str(i) for i in range(1,6+1)] 
	LIST_OF_DATA_NAME = ['positive_mean_power', 'negative_mean_power', 'average_mean_power']
	custom_sequence_abstract_0002(config_data, tracker_name, 
		LIST_OF_LAYERS_TO_OBSERVE,LIST_OF_DATA_NAME,
		tab_level=tab_level, verbose=verbose)

def custom_sequence_abstract_0002(config_data, tracker_name, 
	LIST_OF_LAYERS_TO_OBSERVE,LIST_OF_DATA_NAME,
	tab_level=0, verbose=250):
	from utils.logger import Logger
	import pipeline.drivethru.drivethru_smallnet_mnist as smut

	sm = choose_import_package_by_tracker_name(tracker_name)
	series_code = config_data['console_subsubmode']
	config_data = manage_config_data_by_code(series_code,config_data)

	time_collect = {'drivethru': {},'visual':0}

	state_tracker = smut.setup_state_tracker(config_data, tracker_name, for_training=True, verbose=verbose, tab_level=tab_level)
	full_path_log_file = ut.get_model_folder_path(config_data, state_tracker, tab_level=tab_level+1)
	sys.stdout = Logger(full_path_log_file=full_path_log_file)
	
	NO_OF_RUNS = config_data['NO_OF_RUNS']
	for i in range(1, 1 + NO_OF_RUNS):
		start = time.time()

		choose_drivethru_method_by_tracker_name(tracker_name, sm, config_data)

		end = time.time()
		time_secs = end - start
		tp.print_smh('run_time',start, end, if_x_times=(10.,100.),
			verbose=verbose, tab_level=tab_level, verbose_threshold=None)
		time_collect['drivethru'][i] = time_secs
		ut.print_partition()

	t_vis1 = time.time()
	if series_code is None:
		print('No subsubmode is selected. Not performing drivethru visual evaluation.')
	else:
		arranged_data, state_tracker = arrange_by_iter(tracker_name, config_data, 
			DEBUG_OVERVIEW_ONLY=False, verbose=250, tab_level=tab_level)
		path_to_save_folder = get_path_to_save_folder(state_tracker)
		plot_loss(arranged_data['loss_data_by_epoch'], path_to_save_folder)
		plot_accuracy(arranged_data['accuracy_vs_iter'], path_to_save_folder)
		visualize_mean_power_by_layer_over_iter(arranged_data['processed_datapoint_vs_iter'], 
			path_to_save_folder, LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME, verbose=100, tab_level=0)
		visualize_filter_effect_over_iter(arranged_data['saliency_by_groundtruth_vs_iter'],
			path_to_save_folder, verbose=250, tab_level=0)	
	t_vis2 = time.time()
	time_collect['visual'] = t_vis2 - t_vis1 # seconds
	print_time_collect(time_collect)

	



def manage_config_data_by_code(series_code,config_data):
	if series_code is None:
		NO_OF_RUNS = 4
		config_data['training']['series_name'] = 'Custom_TEST01'
		config_data['visual']['series_name'] = 'Custom_TEST01'
		config_data['drivethru']['no_of_evals_per_run'] = 6 # NO_OF_EVALUATION_DESIRED per run

		config_data['lrp']['relprop_mode'] = 'relprop1'
	else:
		config_data['training']['series_name'] = 'Custom_' + str(series_code)
		config_data['visual']['series_name'] = 'Custom_' + str(series_code)

		# parse series_code here if needed
		# e.g. RXXXX1
		if series_code[0] == 'R': config_data['lrp']['relprop_mode'] = 'relprop1'
		elif series_code[0] == 'L': config_data['lrp']['relprop_mode'] = 'relprop2'

		# NO_OF_RUNS = 4 # revision 1. Need to adjust based on Networks and data
		if series_code[1] == 'X': 
			NO_OF_RUNS = 1
			config_data['drivethru']['no_of_evals_per_run'] = 40 # NO_OF_EVALUATION_DESIRED per run
			config_data['general']['epoch'] = 1
		elif series_code[1] == 'Y': 
			NO_OF_RUNS = 4
			config_data['drivethru']['no_of_evals_per_run'] = 10 # NO_OF_EVALUATION_DESIRED per run
			config_data['general']['epoch'] = 1
		if series_code[1] == 'S': # for testing 
			NO_OF_RUNS = 1
			config_data['drivethru']['no_of_evals_per_run'] = 2 # NO_OF_EVALUATION_DESIRED per run
			config_data['general']['epoch'] = 1
		elif series_code[1] == 'Y': 
			NO_OF_RUNS = 2
			config_data['drivethru']['no_of_evals_per_run'] = 3 # NO_OF_EVALUATION_DESIRED per run
			config_data['general']['epoch'] = 2			
		else:
			raise Exception('select appropriate series_code[1]')
		config_data['NO_OF_RUNS'] = NO_OF_RUNS

		if series_code[2] == 'X':
			pass
		elif series_code[2] == 'M':
			# MNIST ONLY
			config_data['data_from_torch']['mnist']['resize'] = (140,140)
		elif series_code[2] == 'C':
			# CIFAR ONLY
			config_data['data_from_torch']['cifar']['resize'] = (256,256)
		else:
			raise Exception('select appropriate series_code[2]')			

		return config_data


def choose_import_package_by_tracker_name(tracker_name):
	# SPECIFIC OPTIONS
	if tracker_name == 'drivethru_0001_smallnet_mnist':
		import pipeline.drivethru.drivethru_smallnet_mnist as sm
	elif tracker_name == 'drivethru_0001_alexnet_mnist':
		import pipeline.drivethru.drivethru_alexnet_mnist as sm
	elif tracker_name == 'drivethru_0001_vgg_mnist':
		import pipeline.drivethru.drivethru_vgg_mnist as sm
	elif tracker_name == 'drivethru_0001_smallnet_cifar':
		import pipeline.drivethru.drivethru_smallnet_cifar as sm
	elif tracker_name == 'drivethru_0001_alexnet_cifar':
		import pipeline.drivethru.drivethru_alexnet_cifar as sm
	elif tracker_name == 'drivethru_0001_vgg_cifar':
		import pipeline.drivethru.drivethru_vgg_cifar as sm
	else:
		raise RuntimeError('Invalid tracker_name!')
	return sm


def choose_drivethru_method_by_tracker_name(tracker_name, sm, config_data):
	# # SPECIFIC OPTIONS
	if tracker_name == 'drivethru_0001_smallnet_mnist':
		sm.drivethru_0001_smallnet_mnist(config_data, tab_level=0, verbose=250)
	elif tracker_name == 'drivethru_0001_alexnet_mnist':
		sm.drivethru_0001_alexnet_mnist(config_data, tab_level=0, verbose=250)
	elif tracker_name == 'drivethru_0001_vgg_mnist':
		sm.drivethru_0001_vgg_mnist(config_data, tab_level=0, verbose=250)
	elif tracker_name == 'drivethru_0001_smallnet_cifar':
		sm.drivethru_0001_smallnet_cifar(config_data, tab_level=0, verbose=250)
	elif tracker_name == 'drivethru_0001_alexnet_cifar':
		sm.drivethru_0001_alexnet_cifar(config_data, tab_level=0, verbose=250)
	elif tracker_name == 'drivethru_0001_vgg_cifar':
		sm.drivethru_0001_vgg_cifar(config_data, tab_level=0, verbose=250)
	else:
		raise RuntimeError('Invalid tracker_name!')		