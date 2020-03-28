from utils.utils import *
import pipeline.drivethru.drivethru_smallnet_mnist as dtsm  

##################################################################
# drivethru utilities
##################################################################
def get_latest_saved_run(config_data, tracker_name, verbose=250, tab_level=0):
	state_tracker = dtsm.DrivethruTracker(
		mode=None,
		ckpt_path=os.path.join(config_data['working_dir'],'checkpoint'),
		training_series_name=tracker_name + str(config_data['visual']['series_name']),
		load_this_n_th_run=None,
		verbose=verbose,
		tab_level=tab_level)
	latest_saved_run = state_tracker.find_latest_save_file()
	return int(latest_saved_run)

def get_tracker_for_this_run(n_th_run, tracker_name, config_data, verbose, tab_level):
	state_tracker = dtsm.DrivethruTracker(
		mode='load_at_n_th_run',
		ckpt_path=os.path.join(config_data['working_dir'],'checkpoint'),
		training_series_name=tracker_name + str(config_data['visual']['series_name']),
		load_this_n_th_run=n_th_run,
		verbose=verbose,
		tab_level=tab_level)

	state_tracker = state_tracker.load_instance(n_th_run, for_training=False)
	##########################################################	
	# update after loading> This is for when directory has changed
	state_tracker.ckpt_path = os.path.join(config_data['working_dir'],'checkpoint')
	state_tracker.series_folder_path = os.path.join(state_tracker.ckpt_path, state_tracker.training_series_name)
	##########################################################
	return state_tracker


def arrange_by_iter(tracker_name, config_data, 
	DEBUG_OVERVIEW_ONLY=False, verbose=250, tab_level=0):

	latest_saved_run = get_latest_saved_run(config_data, tracker_name,
		verbose=verbose, tab_level=tab_level+1)
	pm.printvm('arrange_by_iter() latest_saved_run:%s'%(str(latest_saved_run)),
		verbose=verbose, verbose_threshold=20, tab_level=tab_level)

	# list_of_lrp_packages_vs_iter = {}
	loss_data_by_epoch = []
	accuracy_vs_iter = {}
	processed_datapoint_vs_iter = {}
	saliency_by_groundtruth_vs_iter = {}

	for n_th_run in range(1,1+latest_saved_run):
		state_tracker = get_tracker_for_this_run(n_th_run, tracker_name, config_data, verbose, tab_level+1)
		data = state_tracker.save_data_by_iter_details
		pm.printvm('%8s | %10s | %28s | %12s'%(str('n_epoch'),str('this_iter'),str('data name'), str('')),
			verbose=verbose, verbose_threshold=300, tab_level=tab_level+1)
		for n_epoch, data_this_epoch in data.items():
			for this_iter, data_this_iter in data_this_epoch.items():
				for data_name, this_data in data_this_iter.items(): 
					pm.printvm('%8s | %10s | %28s | %12s'%(str(n_epoch),str(this_iter), 
						str(data_name), str(type(this_data))),
						verbose=verbose, verbose_threshold=300, tab_level=tab_level+1)
					
					# if not np.all(np.isfinite(this_data)):
					# 	print('WARNING. arrange_by_iter(). Non-finite value in this_data:%s.'%(str(data_name)))

					if DEBUG_OVERVIEW_ONLY: continue
					if data_name == 'accuracy':
						accuracy_vs_iter[this_iter] = this_data
					if data_name == 'list_of_processed_datapoint':
						processed_datapoint_vs_iter[this_iter] = this_data
					if data_name == 'saliency_by_groundtruth':
						saliency_by_groundtruth_vs_iter[this_iter] = this_data
					if data_name == 'data_agg_by_ground_truth':
						raise Exception(DEPRECATE_MESSAGE)
					if data_name == 'list_of_lrp_packages':
						raise Exception(DEPRECATE_MESSAGE)
						# list_of_lrp_packages_vs_iter[this_iter] = this_data
	
	# use latest state_tracker
	for n_epoch in state_tracker.save_data_by_epoch:
		loss_data_by_epoch.append(np.array(state_tracker.save_data_by_epoch[n_epoch]['loss']))
	
	loss_data_by_epoch = np.array(loss_data_by_epoch)
	pm.printvm('loss_data_by_epoch.shape:%s'%(str(loss_data_by_epoch.shape)),
		verbose=250, verbose_threshold=20, tab_level=tab_level+1)		
	
	arranged_data = {
		'accuracy_vs_iter':accuracy_vs_iter,
		'processed_datapoint_vs_iter': processed_datapoint_vs_iter, 
		'saliency_by_groundtruth_vs_iter': saliency_by_groundtruth_vs_iter,
		'loss_data_by_epoch':loss_data_by_epoch
	}
	return arranged_data, state_tracker