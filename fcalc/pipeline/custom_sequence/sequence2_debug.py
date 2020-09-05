from utils.utils import *
import pipeline.custom_sequence.sequence_utils as ut
from utils.logger import TimePrinter
tp = TimePrinter()

from visual.drivethru_visual_smallnet_mnist_functions import *
from visual.drivethru_visual_smallnet_mnist_functions2 import *
from visual.drivethru_visual_smallnet_mnist_preprocessing import *

TOGGLE_TRAINING = False
TOGGLE_VIS = True

def custom_sequence_abstract_0002_debug(config_data, tracker_name, 
	LIST_OF_LAYERS_TO_OBSERVE,LIST_OF_DATA_NAME,
	ORDERED_LIST_OF_GROUNDTRUTH_TO_OBSERVE, LIST_OF_LAYERS_TO_OBSERVE_2, LIST_OF_DATA_NAME_2,
	tab_level=0, verbose=250):

	print('custom_sequence_abstract_0002_debug()')

	from utils.logger import Logger
	import pipeline.drivethru.drivethru_smallnet_mnist as smut

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

	series_code = config_data['console_subsubmode']
	if series_code is None:
		NO_OF_RUNS = 4
		config_data['training']['series_name'] = 'Custom_TEST01'
		config_data['visual']['series_name'] = 'Custom_TEST01'
		config_data['drivethru']['no_of_evals_per_run'] = 6 # NO_OF_EVALUATION_DESIRED per run

		config_data['lrp']['relprop_mode'] = 'relprop1'
	else:
		NO_OF_RUNS = 4 # revision 1. Need to adjust based on Networks and data
		config_data['training']['series_name'] = 'Custom_' + str(series_code)
		config_data['visual']['series_name'] = 'Custom_' + str(series_code)
		config_data['drivethru']['no_of_evals_per_run'] = 25 # NO_OF_EVALUATION_DESIRED per run

		# parse series_code here if needed
		# e.g. RXXXX1
		if series_code[0] == 'R': config_data['lrp']['relprop_mode'] = 'relprop1'
		elif series_code[1] == 'L': config_data['lrp']['relprop_mode'] = 'relprop2'


	# time_collect = {'drivethru': {},'visual':0}

	state_tracker = smut.setup_state_tracker(config_data, tracker_name, for_training=True, verbose=verbose, tab_level=tab_level)
	full_path_log_file = ut.get_model_folder_path(config_data, state_tracker, tab_level=tab_level+1)

	sys.stdout = Logger(full_path_log_file=full_path_log_file)
	
	if TOGGLE_TRAINING:
		for i in range(1, 1 + NO_OF_RUNS):
			start = time.time()

			# SPECIFIC OPTIONS
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

			# end = time.time()
			# time_secs = end - start
			# tp.print_smh('run_time',start, end, if_x_times=(10.,100.),
			# 	verbose=verbose, tab_level=tab_level, verbose_threshold=None)
			# time_collect['drivethru'][i] = time_secs
			# ut.print_partition()


	if TOGGLE_VIS:
			# t_vis1 = time.time()
			# if series_code is None:
			# 	print('No subsubmode is selected. Not performing drivethru visual evaluation.')
			# else:
		arranged_data, state_tracker = arrange_by_iter(tracker_name, config_data, 
			DEBUG_OVERVIEW_ONLY=False, verbose=250, tab_level=tab_level)

		############################################################
		# AT THIS POINT, IF YOU HAVE SENT THE JOB
		# TO SOME GPU CLUSTER, THE FOLLOWING directory MIGHT HAVE TO BE 
		# CHANGED FOR POSTPOST PROCESSING
		# e.g. state_tracker.series_folder_path
		#   /mnt/checkpoint/drivethru_0001_alexnet_mnistCustom_RXXXX1
		#   may not exist in your current machine. So, change it!
		############################################################
		state_tracker.ckpt_path = os.path.join(config_data['working_dir'],'checkpoint')
		state_tracker.series_folder_path = os.path.join(state_tracker.ckpt_path, state_tracker.training_series_name)

		path_to_save_folder = get_path_to_save_folder(state_tracker)
		# plot_loss(arranged_data['loss_data_by_epoch'], path_to_save_folder)
		# plot_accuracy(arranged_data['accuracy_vs_iter'], path_to_save_folder)
		

		# visualize_mean_power_by_layer_over_iter(arranged_data['processed_datapoint_vs_iter'], 
		# 	path_to_save_folder, LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME, verbose=100, tab_level=0)

		##########################################################################
		for LAYER_TO_OBSERVE in LIST_OF_LAYERS_TO_OBSERVE_2:
			visualize_mean_power_by_layer_over_iter2(arranged_data['groundtruth_agg_vs_iter'],
				LAYER_TO_OBSERVE, ORDERED_LIST_OF_GROUNDTRUTH_TO_OBSERVE, LIST_OF_DATA_NAME_2,
				path_to_save_folder, verbose=250, tab_level=0)
		# t_vis2 = time.time()
		# time_collect['visual'] = t_vis2 - t_vis1 # seconds
		# print_time_collect(time_collect)