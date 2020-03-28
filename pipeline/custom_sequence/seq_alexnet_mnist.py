from utils.utils import *
from utils.logger import TimePrinter
from utils.logger import Logger
from pipeline.custom_sequence.sequence2 import print_time_collect
import pipeline.drivethru.drivethru_smallnet_mnist as smut
import pipeline.custom_sequence.sequence_utils as ut

import visual.drivethru_visual_smallnet_mnist_preprocessing as vdp
import visual.drivethru_visual_smallnet_mnist_functions as vd
import visual.drivethru_visual_smallnet_mnist_functions3 as vd3

from pipeline.custom_sequence.seq_smallnet_mnist import running_drivethru, running_visuals

tp = TimePrinter()
def seq_alexnet_mnist_upsize(config_data, tab_level=0, verbose=250):
	print('seq_alexnet_mnist_upsize()')
	config_data['data_from_torch']['mnist']['resize'] = (140,140)

	config_data['NO_OF_RUNS'] = 1
	config_data['general']['epoch'] = 1 # 1 epoch enough
	config_data['drivethru']['eval_every_n_iter'] = 1000
	config_data['drivethru']['n_of_test_data_per_eval'] = 1000
	config_data['drivethru']['n_of_test_data_per_LRP_eval'] = 240	
	seq_alexnet_mnist_general(config_data, tab_level=0, verbose=250)

def seq_alexnet_mnist(config_data, tab_level=0, verbose=250):
	print('seq_alexnet_mnist()')
	config_data['data_from_torch']['mnist']['resize'] = None

	config_data['NO_OF_RUNS'] = 1
	config_data['general']['epoch'] = 1 # 1 epoch enough
	config_data['drivethru']['eval_every_n_iter'] = 1000
	config_data['drivethru']['n_of_test_data_per_eval'] = 1000
	config_data['drivethru']['n_of_test_data_per_LRP_eval'] = 240	
	seq_alexnet_mnist_general(config_data, tab_level=0, verbose=250)

def seq_alexnet_mnist_general(config_data, tab_level=0, verbose=250):
	# model/data specific
	tracker_name = 'drivethru_0001_alexnet_mnist'
	LIST_OF_LAYERS_TO_OBSERVE = ['fn'+str(i) for i in range(1, 3+1)] + ['convb_'+str(i) for i in range(1,5+1)] 
	LIST_OF_DATA_NAME = ['positive_mean_power', 'negative_mean_power', 'average_mean_power']
	import pipeline.drivethru.drivethru_alexnet_mnist as sm
	this_drivethru_method = sm.drivethru_0001_alexnet_mnist


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
	# time_collect = running_drivethru(config_data, this_drivethru_method, time_collect, tracker_name,
	# 	verbose=verbose, tab_level=tab_level)
	time_collect = running_visuals(config_data,LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME, 
		tracker_name, time_collect,	verbose=verbose, tab_level=tab_level)

	print_time_collect(time_collect)
