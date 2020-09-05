from utils.utils import *
from pipeline.training.train_smallnet_mnist import setup_training_tools_0001, \
	save_model_by_n_th_run, get_path_to_save_model, get_path_to_load_model
from pipeline.training.train_smallnet_mnist_debug import DEBUG_train_loop_0001, DEBUG_train_loop_0002

if IS_DEBUG:
	DEBUG_train_smallnet_mnist_LOOP_SIGNAL = 0
	DEBUG_train_smallnet_mnist_LOOP2_SIGNAL = 1
else:
	DEBUG_train_smallnet_mnist_LOOP_SIGNAL = 0
	DEBUG_train_smallnet_mnist_LOOP2_SIGNAL = 0

def train_smallnet_cifar_0001(config_data, tab_level=0, verbose=0):
	print('train_smallnet_cifar_0001()')
	
	from pipeline.data.load_data_cifar import load_cifar_0001
	from utils.loss import compute_loss

	trainloader = load_cifar_0001(config_data, batch_size=None, shuffle=True, verbose=verbose)
	state_tracker = setup_state_tracker(config_data, verbose=verbose, tab_level=tab_level)
	net = new_or_load_model(state_tracker,config_data, verbose=verbose, tab_level=tab_level)

	criterion, optimizer = setup_training_tools_0001(net, config_data, 
		verbose=verbose, tab_level=tab_level+1)

	pm.printv('Start training...'%(), tab_level=tab_level)
	total_iter_in_this_run = 0
	l_epoch = 1 + state_tracker.get_latest_saved_epoch()
	for n_epoch in range(l_epoch, l_epoch + config_data['general']['epoch']):
		state_tracker.setup_for_this_epoch(n_epoch)
		for i, data in enumerate(trainloader,0):
			optimizer.zero_grad()

			x, y0 = data
			x = x.to(this_device)
			if DEBUG_train_loop_0001(DEBUG_train_smallnet_mnist_LOOP_SIGNAL, net, x, y0, 
				tab_level=tab_level, verbose=verbose): return

			y = net(x)

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
	tracker_name = 'smallnet_cifar_0001_'
	
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

def new_or_load_model(state_tracker, config_data, verbose=0, tab_level=0):
	pm.printvm("new_or_load_model(). latest_saved_run:%s"%(str(state_tracker.latest_saved_run)),
		 tab_level=tab_level,verbose=verbose, verbose_threshold=50)
	from models.networks import SmallCNN

	if config_data['data_from_torch']['cifar']['resize'] is None:
		IMG_SIZE = (32,32)
		net = SmallCNN(IMG_SIZE=IMG_SIZE, INPUT_CHANNEL_SIZE =3 ,verbose=verbose, tab_level=tab_level)
	else:
		IMG_SIZE = config_data['data_from_torch']['cifar']['resize']
		net = SmallCNN(IMG_SIZE=IMG_SIZE, INPUT_CHANNEL_SIZE =3 ,set_default_params=False,verbose=verbose, tab_level=tab_level)
		if IMG_SIZE == (256,256):
			from models.networks_smallCNN import SmallCNN_custom_setting0002
			net = SmallCNN_custom_setting0002(net, IMG_SIZE)
		else:
			raise Exception('Please manually set params')

	from pipeline.training.train_smallnet_mnist import new_or_load_model_inner
	net = new_or_load_model_inner(state_tracker, net, verbose=verbose, tab_level=tab_level+1)
	return net

