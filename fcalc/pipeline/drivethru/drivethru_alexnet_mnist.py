from utils.utils import *
from pipeline.data.load_data import load_mnist_0001, AutoReloaderTestMNIST
from utils.loss import compute_loss
import pipeline.drivethru.drivethru_smallnet_mnist as du
import pipeline.drivethru.drivethru_smallnet_mnist_utils as dut
import pipeline.training.train_smallnet_mnist as trsm

if IS_DEBUG:
	DEBUG_N_ITER_MAX_PER_EPOCH = 10
	DEBUG_DRIVE_TRHU_LOOP = 0
else:
	DEBUG_N_ITER_MAX_PER_EPOCH = 0 # (int)
	DEBUG_DRIVE_TRHU_LOOP = False # (bool)

def drivethru_0001_alexnet_mnist(config_data, tab_level=0, verbose=250):
	print('drivethru_0001_alexnet_mnist(). ')
	PROGRESS_TRACK_EVERY_N_PERCENT = 5.

	tracker_name = 'drivethru_0001_alexnet_mnist'
	config_data = dut.set_debug_config(IS_DEBUG, config_data)
	pm.print_recursive_dict(config_data, tab_level=tab_level+2, verbose=verbose, verbose_threshold=None)
	
	data_loader, n_batch_train = dut.setup_training_and_data_loader(config_data, verbose=0, tab_level=0)	
	autoreloader = AutoReloaderTestMNIST(config_data, shuffle=True)
	
	eval_every_n_iter = dut.get_eval_every_n_iter(config_data, n_batch_train, config_data['general']['epoch'], 
		NO_OF_EVALUATION_DESIRED=config_data['drivethru']['no_of_evals_per_run'],
		manual_specification=True, 
		DEBUG_N_ITER_MAX_PER_EPOCH=DEBUG_N_ITER_MAX_PER_EPOCH)

	state_tracker = dut.setup_state_tracker(config_data, tracker_name, for_training=True, verbose=250, tab_level=0)
	net = new_or_load_model(state_tracker, config_data, verbose=verbose, tab_level=tab_level)
	criterion, optimizer = trsm.setup_training_tools_0001(net, config_data, 
		verbose=verbose, tab_level=tab_level+1)	

	save_data_by_iter_details = {} # split save data by runs
	total_iter_in_this_run = 0
	total_global_iter = state_tracker.save_data_by_nth_run[state_tracker.current_run]['total_iteration']
	last_saved_epoch = 1 + state_tracker.get_latest_saved_epoch()
	progress_tracker = int(n_batch_train/(100/PROGRESS_TRACK_EVERY_N_PERCENT))

	pm.printv('Start drive through...'%(), tab_level=tab_level)
	for n_epoch in range(last_saved_epoch, last_saved_epoch + config_data['general']['epoch']):
		state_tracker.setup_for_this_epoch(n_epoch, tab_level=tab_level+1, verbose=verbose)
		for i, data in enumerate(data_loader,0):
			dut.print_progress_percentage(i, progress_tracker, n_batch_train, verbose=250, tab_level=tab_level+1)
			
			optimizer.zero_grad()

			x, y0 = data
			x = x.to(this_device)
			net.train()
			y = net(x)

			loss = compute_loss(criterion, y.squeeze(3).squeeze(2).cpu(), y0)
			loss.backward()
			optimizer.step()

			# Drive through LRP and evaluation
			if (total_iter_in_this_run+1)%eval_every_n_iter == 0:
				save_data_by_iter_details, autoreloader = du.drive_thru_evaluation(
					net, autoreloader, x, y, y0, save_data_by_iter_details,
					i, total_global_iter + total_iter_in_this_run, n_epoch, 
					n_of_test_data_per_eval=config_data['drivethru']['n_of_test_data_per_eval'], 
					n_of_test_data_per_LRP_eval=config_data['drivethru']['n_of_test_data_per_LRP_eval'],
					DEBUG_DRIVE_TRHU_LOOP = 0, # DEBUG not available
					tab_level=tab_level+1, verbose=verbose)
				# just_in_time_display(save_data_by_iter_details, verbose=verbose, tab_level=tab_level+1)
				if DEBUG_DRIVE_TRHU_LOOP: return


			# FOR LOGGING
			total_iter_in_this_run += 1
			state_tracker.store_loss_by_epoch(loss.item(), n_epoch)

			stop_iter, stop_epoch = dut.DEBUG_train_loop_0002(DEBUG_N_ITER_MAX_PER_EPOCH, 
				i, n_epoch - last_saved_epoch , tab_level=tab_level+1, verbose=verbose)		
			if stop_iter: break
		# if DEBUG_DRIVE_TRHU_LOOP2: return
		state_tracker.update_epoch()
		if stop_epoch: break

	state_tracker.save_data_by_iter_details = save_data_by_iter_details
	state_tracker.update_state(total_iter_in_this_run, config_data)
	trsm.save_model_by_n_th_run(net, state_tracker,tab_level=tab_level, verbose=verbose)
	state_tracker.display_end_state(tab_level=tab_level+1, verbose=verbose)	

def new_or_load_model(state_tracker, config_data, verbose=0, tab_level=0):
	from models.networks import AlexNetLikeprobe0001
	
	if config_data['data_from_torch']['mnist']['resize'] is None:
		net = AlexNetLikeprobe0001(relprop_mode=config_data['lrp']['relprop_mode'], verbose=verbose, tab_level=tab_level)
	else:
		IMG_SIZE = config_data['data_from_torch']['mnist']['resize']
		net = AlexNetLikeprobe0001(set_default_params=False, relprop_mode=config_data['lrp']['relprop_mode'], verbose=verbose, tab_level=tab_level)
		if IMG_SIZE == (140,140):
			from models.networks_Alex import AlexNet_custom_setting0001 
			net = AlexNet_custom_setting0001(net)
		else:
			raise Exception('Please manually set params')
	net = dut.new_or_load_model_inner(state_tracker, net, verbose=verbose, tab_level=tab_level)
	return net

