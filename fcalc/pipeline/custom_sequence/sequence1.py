from utils.utils import *
import pipeline.custom_sequence.sequence_utils as ut
from utils.logger import TimePrinter
tp = TimePrinter()

def custom_sequence_0001(config_data, tab_level=0, verbose=250):
	print('custom_sequence_0001()')

	import pipeline.custom_sequence.sequence1_functions as s1
	import pipeline.training.train_smallnet_mnist as tr
	from utils.logger import Logger

	##########################################################
	N_TRAINING_RUN = 3
	N_EPOCH_PER_RUN = 2
	DO_TRAINING = 1
	DO_EVAL1 = 1
	DO_EVAL2 = 1

	##########################################################	
	state_tracker = tr.setup_state_tracker(config_data, verbose=verbose, tab_level=tab_level)
	full_path_log_file = ut.get_model_folder_path(config_data, state_tracker, tab_level=tab_level+1)
	sys.stdout = Logger(full_path_log_file=full_path_log_file)

	##########################################################
	time_collect = {
		'training':{},
		'evaluation':{},
		'lrp':{}
	}
	if DO_TRAINING:
		config_data['general']['epoch'] = N_EPOCH_PER_RUN 
		for run_number in range(1 ,1+N_TRAINING_RUN):
			time_collect['training'][run_number] = s1.train_smallnet_mnist_0001_timed(run_number, config_data, 
				tab_level=tab_level, verbose=verbose)
	if DO_EVAL1:
		time_collect['evaluation']['overfit'] = s1.eval_smallnet_mnist_0001_overfitting_timed(config_data, tab_level=0, verbose=250)
		 
	if DO_EVAL2:
		time_collect['evaluation']['test'] = s1.eval_smallnet_mnist_0002_test_timed(config_data, tab_level=0, verbose=250)
		
	print_time_collect(time_collect, tab_level=0)


def print_time_collect(time_collect, tab_level=0):
	print('print_time_collect().')
	for run_number, time_secs in time_collect['training'].items():
		tp.print_smh('training_'+ str(run_number), 0, time_secs, if_x_times=(10.,100.),
			verbose=0, tab_level=1, verbose_threshold=None)		
	tp.print_smh('eval_overfit', 0, time_collect['evaluation']['overfit'], if_x_times=(10.,100.),
		verbose=0, tab_level=1, verbose_threshold=None)		
	tp.print_smh('eval_overfit', 0, time_collect['evaluation']['test'], if_x_times=(10.,100.),
		verbose=0, tab_level=1, verbose_threshold=None)		
		