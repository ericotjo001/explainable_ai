from utils.utils import *
from pipeline.training.train_smallnet_mnist_debug import *

def train_smallnet_mnist_0001(config_data, tab_level=0, verbose=0):
	print('train_smallnet_mnist_0001().')

	from pipeline.data.load_data import load_mnist_0001
	from utils.loss import compute_loss

	data_loader = load_mnist_0001(config_data, verbose=0)
	state_tracker = setup_state_tracker(config_data, verbose=verbose, tab_level=tab_level)
	net = new_or_load_model(state_tracker, config_data, verbose=verbose, tab_level=tab_level)

	criterion, optimizer = setup_training_tools_0001(net, config_data, 
		verbose=verbose, tab_level=tab_level+1)

	pm.printv('Start training...'%(), tab_level=tab_level)
	total_iter_in_this_run = 0
	l_epoch = 1 + state_tracker.get_latest_saved_epoch()
	for n_epoch in range(l_epoch, l_epoch + config_data['general']['epoch']):
		state_tracker.setup_for_this_epoch(n_epoch)
		for i, data in enumerate(data_loader,0):
			optimizer.zero_grad()

			x, y0 = data
			if DEBUG_train_loop_0001(DEBUG_train_smallnet_mnist_LOOP_SIGNAL, net, x, y0, 
				tab_level=tab_level, verbose=verbose): return

			y = net(x.to(this_device))

			loss = compute_loss(criterion, y.squeeze(3).squeeze(2).cpu(), y0)
			loss.backward()
			optimizer.step()

			# FOR LOGGING
			total_iter_in_this_run += 1
			state_tracker.store_loss_by_epoch(loss.item(), n_epoch)

			stop_iter, stop_epoch = DEBUG_train_loop_0002(DEBUG_train_smallnet_mnist_LOOP2_SIGNAL, 
				i, n_epoch - l_epoch , tab_level=tab_level+1, verbose=verbose)		
			if stop_iter: break
		state_tracker.update_epoch()
		if stop_epoch: break
	state_tracker.update_state(total_iter_in_this_run, config_data)
	save_model_by_n_th_run(net, state_tracker,tab_level=tab_level, verbose=verbose)
	state_tracker.display_end_state(tab_level=tab_level+1, verbose=verbose)

def setup_state_tracker(config_data, for_training=True, verbose=0, tab_level=0):
	tracker_name = 'smallnet_mnist_0001_'
	
	from utils.state_tracker import StateTracker 
	ckpt_path = os.path.join(config_data['working_dir'],'checkpoint')
	training_series_name = tracker_name + str(config_data['training']['series_name'])
	state_tracker = StateTracker(
		mode='load_latest_if_available',
		ckpt_path=ckpt_path,
		training_series_name=training_series_name,
		load_this_n_th_run=None,
		verbose=verbose,
		tab_level=tab_level+1)
	state_tracker = state_tracker.load_latest_if_available(state_tracker.latest_saved_run,for_training=for_training)
	state_tracker.display_simple_state(tab_level=tab_level+1, verbose=verbose)	
	return state_tracker

def new_or_load_model(state_tracker, config_data ,verbose=0, tab_level=0):
	pm.printvm("new_or_load_model(). latest_saved_run:%s"%(str(state_tracker.latest_saved_run)),
		 tab_level=tab_level,verbose=verbose, verbose_threshold=50)
	from models.networks import SmallCNN

	if config_data['data_from_torch']['mnist']['resize'] is None:
		net = SmallCNN(verbose=verbose, tab_level=tab_level)
	else:
		IMG_SIZE = config_data['data_from_torch']['mnist']['resize']
		net = SmallCNN(set_default_params=False,verbose=verbose, tab_level=tab_level)
		if IMG_SIZE == (140,140):
			from models.networks_smallCNN import SmallCNN_custom_setting0001
			net = SmallCNN_custom_setting0001(net, IMG_SIZE)
		else:
			raise Exception('Please manually set params')
	net = new_or_load_model_inner(state_tracker, net, verbose=verbose, tab_level=tab_level+1)
	return net

def new_or_load_model_inner(state_tracker, net, verbose=0, tab_level=0):
	pm.printvm("new_or_load_model_inner()"%(),
		tab_level=tab_level,verbose=verbose, verbose_threshold=50)

	if state_tracker.latest_saved_run is None:
		pm.printvm("Init new model!"%(),
			tab_level=tab_level+1,verbose=verbose, verbose_threshold=50)
	else:
		pm.printvm("Loading model... prev_run:%s"%(str(state_tracker.current_run-1)),
			tab_level=tab_level+1,verbose=verbose, verbose_threshold=50)
		net.load_state_dict(torch.load(get_path_to_load_model(state_tracker, 
			verbose=verbose, tab_level=tab_level+1)))
	return net


#############################################################################
# Abstracted functions (no model specified)
#############################################################################

def setup_training_tools_0001(this_net, config_data, verbose=0, tab_level=0):
	pm.printvm("setup_training_tools_0001()."%(),
		 tab_level=tab_level,verbose=verbose, verbose_threshold=50)
	
	criterion = nn.CrossEntropyLoss()
	from utils.optimizer import get_optimizer
	optimizer = get_optimizer(this_net, config_data, verbose=verbose, tab_level=tab_level+1)
	return criterion, optimizer

def get_path_to_load_model(state_tracker, verbose=0, tab_level=0):
	pm.printvm("get_path_to_load_model():"%(),
		tab_level=tab_level,verbose=verbose, verbose_threshold=50)
	prev_run = state_tracker.current_run-1
	model_name = str(state_tracker.training_series_name) + '.' + str(prev_run) + '.model'
	pm.printvm("Loaded path:%s"%(str(state_tracker.series_folder_path)),
		tab_level=tab_level+1,verbose=verbose, verbose_threshold=50)
	pm.printvm("Loaded model:%s"%(str(model_name)),
		tab_level=tab_level+1,verbose=verbose, verbose_threshold=50)	
	return os.path.join(state_tracker.series_folder_path, model_name)

def get_path_to_save_model(state_tracker, verbose=0, tab_level=0):
	model_name = str(state_tracker.training_series_name) + '.' + str(state_tracker.current_run) + '.model'
	pm.printvm("Saved model:%s"%(str(model_name)),
		tab_level=tab_level+1,verbose=verbose, verbose_threshold=50)	
	return os.path.join(state_tracker.series_folder_path, model_name)

def save_model_by_n_th_run(net, state_tracker, tab_level=0, verbose=0):
	torch.save(net.state_dict(), get_path_to_save_model(state_tracker, verbose=verbose, tab_level=tab_level))


