from pipeline.drivethru.drivethru_smallnet_mnist_utils import *

if IS_DEBUG:
	DEBUG_DRIVE_TRHU_LOOP = 0
	DEBUG_DRIVE_TRHU_LOOP3 = 0
	DEBUG_COMPUTE_NUMBER_OF_DRIVETHRU_SIGNAL = 0
	DEBUG_N_ITER_MAX_PER_EPOCH = 10
else:
	DEBUG_DRIVE_TRHU_LOOP = False # (bool)
	DEBUG_DRIVE_TRHU_LOOP3 = False # (bool)
	DEBUG_COMPUTE_NUMBER_OF_DRIVETHRU_SIGNAL = False # (bool)
	DEBUG_N_ITER_MAX_PER_EPOCH = 0 # (int)

EMERGENCY_DEBUG = 0
if EMERGENCY_DEBUG: 
	EMERGENCY_DRIVETHRU_LOOP_SIGNAL = 1
else:
	EMERGENCY_DRIVETHRU_LOOP_SIGNAL = 0

def drivethru_0001_smallnet_mnist(config_data, tab_level=0, verbose=250):
	print('drivethru_0001_smallnet_mnist(). ')
	
	if EMERGENCY_DEBUG: print(EMERGENCY_DEBUG_MSG)
	PROGRESS_TRACK_EVERY_N_PERCENT = 5.

	tracker_name = 'drivethru_0001_smallnet_mnist'
	config_data = set_debug_config(IS_DEBUG, config_data)
	pm.print_recursive_dict(config_data, tab_level=tab_level+2, verbose=verbose, verbose_threshold=None)
	
	data_loader, n_batch_train = setup_training_and_data_loader(config_data, verbose=0, tab_level=0)	
	autoreloader = AutoReloaderTestMNIST(config_data, shuffle=True)
	
	eval_every_n_iter = get_eval_every_n_iter(config_data, n_batch_train, config_data['general']['epoch'], 
		NO_OF_EVALUATION_DESIRED=config_data['drivethru']['no_of_evals_per_run'],
		manual_specification=True, 
		DEBUG_N_ITER_MAX_PER_EPOCH=DEBUG_N_ITER_MAX_PER_EPOCH)
	if DEBUG_COMPUTE_NUMBER_OF_DRIVETHRU(DEBUG_COMPUTE_NUMBER_OF_DRIVETHRU_SIGNAL,
		eval_every_n_iter, n_batch_train, config_data['general']['epoch']): return


	state_tracker = setup_state_tracker(config_data, tracker_name, for_training=True, verbose=250, tab_level=0)
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
			print_progress_percentage(i, progress_tracker, n_batch_train, verbose=250, tab_level=tab_level+1)
			if emergency_drivethru_loop(EMERGENCY_DRIVETHRU_LOOP_SIGNAL, total_iter_in_this_run, eval_every_n_iter): total_iter_in_this_run += 1; continue

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
				save_data_by_iter_details, autoreloader = drive_thru_evaluation(
					net, autoreloader, x, y, y0, save_data_by_iter_details,
					i, total_global_iter + total_iter_in_this_run, n_epoch, 
					n_of_test_data_per_eval=config_data['drivethru']['n_of_test_data_per_eval'], 
					n_of_test_data_per_LRP_eval=config_data['drivethru']['n_of_test_data_per_LRP_eval'],
					DEBUG_DRIVE_TRHU_LOOP = DEBUG_DRIVE_TRHU_LOOP,
					tab_level=tab_level+1, verbose=verbose)
				if DEBUG_DRIVE_TRHU_LOOP: return
				if DEBUG_DRIVE_TRHU_LOOP3: return
				just_in_time_display(save_data_by_iter_details, verbose=0, tab_level=tab_level+1)

			# FOR LOGGING
			total_iter_in_this_run += 1
			state_tracker.store_loss_by_epoch(loss.item(), n_epoch)

			stop_iter, stop_epoch = DEBUG_train_loop_0002(DEBUG_N_ITER_MAX_PER_EPOCH, 
				i, n_epoch - last_saved_epoch , tab_level=tab_level+1, verbose=verbose)		
			if stop_iter: break
		# if DEBUG_DRIVE_TRHU_LOOP2: return
		state_tracker.update_epoch()
		if stop_epoch: break

	state_tracker.save_data_by_iter_details = save_data_by_iter_details
	state_tracker.update_state(total_iter_in_this_run, config_data)
	trsm.save_model_by_n_th_run(net, state_tracker,tab_level=tab_level, verbose=verbose)
	state_tracker.display_end_state(tab_level=tab_level+1, verbose=verbose)	

	
def drive_thru_evaluation(net, autoreloader, x, y, y0, save_data_by_iter_details, i, total_iter, n_epoch, 
	n_of_test_data_per_eval=100, 
	n_of_test_data_per_LRP_eval=10, 
	DEBUG_DRIVE_TRHU_LOOP = 0,
	tab_level=0, verbose=250):
	"""
	Perform evaluation in midst training
	"""
	net.eval()
	from utils.metrics import ClassAccuracy
	Acc = ClassAccuracy()

	pm.printvm('Performing drive-thru evaluation:\n  epoch:%s | iter:%s | total iter so far:%s (one-based)'%(
		str(n_epoch+1),str(i+1),str(total_iter+1)),
		tab_level=tab_level,verbose=verbose, verbose_threshold=50)

	list_of_processed_datapoint = []
	saliency_by_groundtruth = {}
	for j in range(n_of_test_data_per_eval):
		x_test, y0_test = autoreloader.fetch_next_item()

		# accuracy evaluation
		y_test = net(x_test.to(this_device))
		y1 = y_test.squeeze(3).squeeze(2)
		y2 = torch.argmax(y1.squeeze(1)).clone().detach().cpu().numpy()
		Acc.update_acc(int(y2),int(y0_test))

		# LRP evaluation
		if j <= n_of_test_data_per_LRP_eval:
			lrp_package = net.relprop_drivethru0001(x_test.to(this_device), y_test, y0_test.to(this_device))

			if DEBUG_DRIVE_TRHU_LOOP: 
				processed_datapoint = net.process_drivethru_data_0001_debug(lrp_package, verbose=verbose, tab_level=tab_level+1)
				return None, None

			if DEBUG_DRIVE_TRHU_LOOP3:
				net.relprop_drivethru0003_debug(x_test.to(this_device), y_test, y0_test.to(this_device), alpha1=0,alpha2=0.9)
				# net.relprop_drivethru0004_debug(x_test.to(this_device), y_test, y0_test.to(this_device), amplifier=1.2)
				print('y0_test:',y0_test.item())
				print('y2: %s'%(str(int(y2)),))
				return None, None

			processed_datapoint = net.process_drivethru_data_0001(lrp_package)
			lrp_filtered_saliency_package = {
				'main':(x_test.squeeze(0).detach().cpu().numpy(),y0_test.item(),int(y2)),
				'R': lrp_package['R'].squeeze(0).detach().cpu().numpy(),
				'c0.6': net.relprop_drivethru0003(x_test.to(this_device), y_test, y0_test.to(this_device), 
					modifier={'mode':'clamp','alphas':(0.,0.6)}).squeeze(0).detach().cpu().numpy(),
				'c0.2': net.relprop_drivethru0003(x_test.to(this_device), y_test, y0_test.to(this_device), 
					modifier={'mode':'clamp','alphas':(0.,0.2)}).squeeze(0).detach().cpu().numpy(),
				'c0.05': net.relprop_drivethru0003(x_test.to(this_device), y_test, y0_test.to(this_device), 
					modifier={'mode':'clamp','alphas':(0.,0.05)}).squeeze(0).detach().cpu().numpy(),
				'f0.6': net.relprop_drivethru0003(x_test.to(this_device), y_test, y0_test.to(this_device), 
					modifier={'mode':'pass','alphas':(0.,0.6)}).squeeze(0).detach().cpu().numpy(),
				'f0.2': net.relprop_drivethru0003(x_test.to(this_device), y_test, y0_test.to(this_device), 
					modifier={'mode':'pass','alphas':(0.,0.2)}).squeeze(0).detach().cpu().numpy(),
				'f0.05': net.relprop_drivethru0003(x_test.to(this_device), y_test, y0_test.to(this_device), 
					modifier={'mode':'pass','alphas':(0.,0.05)}).squeeze(0).detach().cpu().numpy(),
				'p0.5': net.relprop_drivethru0003(x_test.to(this_device), y_test, y0_test.to(this_device), 
					modifier={'mode':'p_amp','alphas':(0.5,2.)}).squeeze(0).detach().cpu().numpy(),
				'p0.7': net.relprop_drivethru0003(x_test.to(this_device), y_test, y0_test.to(this_device), 
					modifier={'mode':'p_amp','alphas':(0.7,2.)}).squeeze(0).detach().cpu().numpy(),
			}
			list_of_processed_datapoint.append(processed_datapoint)
			if y0_test.item() not in saliency_by_groundtruth:
				saliency_by_groundtruth[y0_test.item()] = []
			saliency_by_groundtruth[y0_test.item()].append(lrp_filtered_saliency_package)

	Acc.compute_acc()
	Acc.display_stats(tab_level=tab_level+2, verbose=verbose,verbose_threshold=250)

	# epochs and iters are all presented 1-based, i.e. does not start from zero
	if n_epoch+1 not in save_data_by_iter_details:
		save_data_by_iter_details[n_epoch+1] = {}
	if total_iter+1 not in save_data_by_iter_details[n_epoch+1]:
		save_data_by_iter_details[n_epoch+1][total_iter+1] = {}
	save_data_by_iter_details[n_epoch+1][total_iter+1]['accuracy'] = Acc.acc
	save_data_by_iter_details[n_epoch+1][total_iter+1]['list_of_processed_datapoint'] = list_of_processed_datapoint 
	save_data_by_iter_details[n_epoch+1][total_iter+1]['saliency_by_groundtruth'] = saliency_by_groundtruth 
	return save_data_by_iter_details, autoreloader

def emergency_drivethru_loop(EMERGENCY_DRIVETHRU_LOOP_SIGNAL, total_iter_in_this_run, eval_every_n_iter):
	EMERGENCY_SIGNAL = False
	if EMERGENCY_DRIVETHRU_LOOP_SIGNAL:
		if not (total_iter_in_this_run+1)%eval_every_n_iter == 0:
			EMERGENCY_SIGNAL = True
	return EMERGENCY_SIGNAL	

def just_in_time_display(save_data_by_iter_details, verbose=250, tab_level=0):
	data = save_data_by_iter_details
	pm.printvm('%8s | %10s | %28s | %12s'%(str('n_epoch'),str('this_iter'),str('data name'), str('')),
		verbose=250, verbose_threshold=20, tab_level=tab_level+1)
	for n_epoch, data_this_epoch in data.items():
		for this_iter, data_this_iter in data_this_epoch.items():
			for data_name, this_data in data_this_iter.items(): 
				pm.printvm('%8s | %10s | %28s | %12s'%(str(n_epoch),str(this_iter), 
					str(data_name), str(type(this_data))),
					verbose=verbose, verbose_threshold=20, tab_level=tab_level)