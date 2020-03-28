from utils.utils import *
import pipeline.lrp.lrp_smallnet_mnist as sm

if IS_DEBUG:
	DEBUG_LOOP1_SIGNAL = 0
	DEBUG_LOOP2_SIGNAL = 10
else:
	DEBUG_LOOP1_SIGNAL = False
	DEBUG_LOOP2_SIGNAL = 0

def lrp_alexnet_mnist_0001(config_data, tab_level=0, verbose=0):
	N_th_RUN_TO_LOAD = config_data['lrp']['n_th_run_to_load'] #  model trained after n-th run 
	config_data['data_from_torch']['mnist']['training_mode'] = True # overfitting
	relprop_mode = config_data['lrp']['relprop_mode']

	state_tracker = prepare_state_tracker_for_use(N_th_RUN_TO_LOAD, config_data, 
		verbose=verbose, tab_level=tab_level)
	net = load_network_for_use(state_tracker,verbose=verbose, tab_level=tab_level)

	sm.lrp_mnist_000X(net, state_tracker,config_data, relprop_mode, N_th_RUN_TO_LOAD,
		DEBUG_LOOP1_SIGNAL, DEBUG_LOOP2_SIGNAL, tab_level=tab_level, verbose=verbose)

def prepare_state_tracker_for_use(N_th_RUN_TO_LOAD, config_data, verbose=0, tab_level=0):
	tracker_name = 'alexnet_mnist_0001_'
	state_tracker = sm.prepare_abstract_state_tracker_for_use(
		N_th_RUN_TO_LOAD, config_data, tracker_name, 
		verbose=verbose, tab_level=tab_level)	
	return state_tracker

def load_network_for_use(state_tracker,verbose=0, tab_level=0):
	from models.networks import AlexNetLike
	net = AlexNetLike(verbose=verbose, tab_level=tab_level)
	net.load_state_dict(torch.load(sm.get_path_to_load_model(state_tracker, 
		verbose=verbose, tab_level=tab_level+1)))
	net.eval()
	return net