from utils.utils import *
from pipeline.training.train_vgg_mnist_debug import *
import pipeline.training.train_smallnet_mnist as mn

def train_vgg_mnist_0001(config_data, tab_level=0, verbose=250):
	print('train_vgg_mnist.py. train_vgg_mnist_0001()')
	from pipeline.data.load_data import load_mnist_0001
	from utils.loss import compute_loss

	data_loader = load_mnist_0001(config_data, verbose=0)
	state_tracker = setup_state_tracker(config_data, verbose=verbose, tab_level=tab_level)
	net = new_or_load_model(state_tracker, verbose=verbose, tab_level=tab_level)
	criterion, optimizer = mn.setup_training_tools_0001(net, config_data, 
		verbose=verbose, tab_level=tab_level+1)

	pm.printv('Start training...'%(), tab_level=tab_level)
	total_iter_in_this_run = 0
	l_epoch = 1 + state_tracker.get_latest_saved_epoch()

	for n_epoch in range(l_epoch, l_epoch + config_data['general']['epoch']):
		state_tracker.setup_for_this_epoch(n_epoch)
		for i, data in enumerate(data_loader,0):
			optimizer.zero_grad()

			x, y0 = data
			x = x.to(this_device)
			if DEBUG_train_loop_0001(DEBUG_train_LOOP_SIGNAL, net, x, y0, 
				tab_level=tab_level, verbose=verbose): return	
			
			y = net(x)

			loss = compute_loss(criterion, y.squeeze(3).squeeze(2).cpu(), y0)
			loss.backward()
			optimizer.step()

			# FOR LOGGING
			total_iter_in_this_run += 1
			state_tracker.store_loss_by_epoch(loss.item(), n_epoch)

			stop_iter, stop_epoch = DEBUG_train_loop_0002(DEBUG_train_LOOP_SIGNAL2, 
				i, n_epoch - l_epoch , tab_level=tab_level+1, verbose=verbose)		
			if stop_iter: break
		state_tracker.update_epoch()
		if stop_epoch: break
	state_tracker.update_state(total_iter_in_this_run, config_data)
	mn.save_model_by_n_th_run(net, state_tracker,tab_level=tab_level, verbose=verbose)
	state_tracker.display_end_state(tab_level=tab_level+1, verbose=verbose)

def setup_state_tracker(config_data, for_training=True, verbose=0, tab_level=0):
	from utils.state_tracker import StateTracker 
	ckpt_path = os.path.join(config_data['working_dir'],'checkpoint')
	tracker_name = 'VGGLike_mnist_0001_'
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

def new_or_load_model(state_tracker, verbose=0, tab_level=0):
	pm.printvm("new_or_load_model(). latest_saved_run:%s"%(str(state_tracker.latest_saved_run)),
		 tab_level=tab_level,verbose=verbose, verbose_threshold=50)
	from models.networks import VGGLike

	net = VGGLike(verbose=verbose, tab_level=tab_level)
	if state_tracker.latest_saved_run is None:
		pm.printvm("Init new model!"%(),
			tab_level=tab_level+1,verbose=verbose, verbose_threshold=50)
	else:
		pm.printvm("Loading model... prev_run:%s"%(str(state_tracker.current_run-1)),
			tab_level=tab_level+1,verbose=verbose, verbose_threshold=50)
		net.load_state_dict(torch.load(mn.get_path_to_load_model(state_tracker, 
			verbose=verbose, tab_level=tab_level+1)))
	return net