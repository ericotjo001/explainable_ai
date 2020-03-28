from utils.utils import *
from pipeline.lrp.lrp_smallnet_mnist_debug import *
from pipeline.data.load_data import load_mnist_0001
from utils.result_wrapper import ResultWrapper
import utils.lrp_utils as lu

#############################################################################
# Smallnet-specific functions for mnist (no specific model used)
#############################################################################

def lrp_smallnet_mnist_0001(config_data, tab_level=0, verbose=0):
	print('lrp_smallnet_mnist_0001(). HARDCODED VARIABLES? YES.')
	relprop_mode = config_data['lrp']['relprop_mode']
	lrp_smallnet_mnist_000X(config_data, relprop_mode, 
		DEBUG_LOOP1_SIGNAL, DEBUG_LOOP2_SIGNAL, tab_level=tab_level, verbose=verbose)

def lrp_smallnet_mnist_000X(config_data, relprop_mode, 
		DEBUG_LOOP1_SIGNAL, DEBUG_LOOP2_SIGNAL, tab_level=0, verbose=0):
	N_th_RUN_TO_LOAD = config_data['lrp']['n_th_run_to_load'] #  model trained after n-th run 
	config_data['data_from_torch']['mnist']['training_mode'] = True # overfitting

	state_tracker = prepare_state_tracker_for_use(N_th_RUN_TO_LOAD, config_data, 
		verbose=verbose, tab_level=tab_level)
	net = load_network_for_use(state_tracker,verbose=verbose, tab_level=tab_level)
	
	lrp_mnist_000X(net, state_tracker,config_data, relprop_mode, N_th_RUN_TO_LOAD,
		DEBUG_LOOP1_SIGNAL, DEBUG_LOOP2_SIGNAL, tab_level=tab_level, verbose=verbose)

def prepare_state_tracker_for_use(N_th_RUN_TO_LOAD, config_data, verbose=0, tab_level=0):
	tracker_name = 'smallnet_mnist_0001_'
	state_tracker = prepare_abstract_state_tracker_for_use(
		N_th_RUN_TO_LOAD, config_data, tracker_name, 
		verbose=verbose, tab_level=tab_level)	
	return state_tracker

def load_network_for_use(state_tracker,verbose=0, tab_level=0):
	from models.networks import SmallCNN
	net = SmallCNN(verbose=verbose, tab_level=tab_level)
	net.load_state_dict(torch.load(get_path_to_load_model(state_tracker, 
		verbose=verbose, tab_level=tab_level+1)))
	net.eval()
	return net

#############################################################################
# Abstracted functions for mnist (no specific model used)
#############################################################################

def lrp_mnist_000X(net, state_tracker, config_data, relprop_mode, N_th_RUN_TO_LOAD,
	DEBUG_LOOP1_SIGNAL, DEBUG_LOOP2_SIGNAL,
	tab_level=0, verbose=0):
	N_SAVE_LRP = 10 # how many data points to LRP
	RELPROP_MODE = relprop_mode

	data_loader = load_mnist_0001(config_data, shuffle=True, batch_size=1, verbose=0)
	net.relprop_mode = RELPROP_MODE
	res = ResultWrapper()
	res.lrp_output = []

	pm.printvm('Start lrp processing...',
		tab_level=tab_level,verbose=verbose, verbose_threshold=250)

	for i, data in enumerate(data_loader,0):
		x, y0 = data
		x = x.to(this_device)

		y = net(x)
		net.forward_lrp(x)
		
		if DEBUG_LOOP1(i,x,y,y0,net, DEBUG_LOOP1_SIGNAL,tab_level=tab_level+1): break
		R = net.relprop(y)
		if DEBUG_LOOP2(i, x, y, y0, R, net, DEBUG_LOOP2_SIGNAL,tab_level=tab_level+1): break
		
		res.lrp_output.append(compactify_result(x,y, y0,R))
		if i>=N_SAVE_LRP: break
	
	working_dir = config_data['working_dir']
	path_to_folder = os.path.join('checkpoint',state_tracker.training_series_name)
	filename = str(RELPROP_MODE) + '_' + str(N_th_RUN_TO_LOAD) + '.result'
	res.save_result(working_dir, path_to_folder, filename, 
		verbose=250, tab_level=0,verbose_threshold=50)

def prepare_abstract_state_tracker_for_use(N_th_RUN_TO_LOAD, config_data, tracker_name, verbose=0, tab_level=0):
	from utils.state_tracker import StateTracker 
	ckpt_path = os.path.join(config_data['working_dir'],'checkpoint')
	training_series_name = str(tracker_name) + str(config_data['training']['series_name'])
	state_tracker = StateTracker(
		mode='load_at_n_th_run',
		load_this_n_th_run=N_th_RUN_TO_LOAD,
		ckpt_path=ckpt_path,
		training_series_name=training_series_name,
		verbose=verbose,
		tab_level=tab_level+1)
	st = state_tracker.load_instance(N_th_RUN_TO_LOAD, for_training=False)
	st.display_end_state(tab_level=tab_level+1, verbose=verbose)
	return state_tracker

def compactify_result(x,y,y0,R):
	R = lu.force_fit_size_x_to_y_2D(R, x)
	return (x.clone().detach().cpu().numpy(),
		y.clone().detach().cpu().numpy(),
		y0.clone().detach().cpu().numpy(),
		R.clone().detach().cpu().numpy())


def get_path_to_load_model(state_tracker, verbose=0, tab_level=0):
	model_name = str(state_tracker.training_series_name) + '.' + str(state_tracker.current_run) + '.model'
	pm.printvm("get_path_to_load_model():\n  Loaded model:%s"%(str(model_name)),
		tab_level=tab_level,verbose=verbose, verbose_threshold=50)
	return os.path.join(state_tracker.series_folder_path, model_name)