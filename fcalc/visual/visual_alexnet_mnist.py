from utils.utils import *
from utils.result_wrapper import ResultWrapper
import visual.visual_small_net_mnist as vsm

def visual_final_LRP_output(config_data, tab_level=0, verbose=0):
	print('visual_alexnet_mnist.py. visual_final_LRP_output(). HARDCODED VARIABLES? YES.')
	SHOW_CASE_INDEX = slice(0,10,None)
	N_th_RUN_TO_LOAD = config_data['visual']['n_th_run_to_load']

	from pipeline.lrp.lrp_smallnet_mnist import prepare_state_tracker_for_use 
	state_tracker = prepare_state_tracker_for_use(N_th_RUN_TO_LOAD, config_data, verbose=verbose, tab_level=tab_level)
	
	vsm.visual_mnist_lrp_abstract_model(config_data, SHOW_CASE_INDEX, N_th_RUN_TO_LOAD,
		state_tracker, tab_level=0, verbose=250)