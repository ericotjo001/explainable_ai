from utils.utils import *

def test_tracker(config_data):
	print('test_tracker()')
	config_data
	state_tracker = setup_state_tracker(config_data, verbose=9999, tab_level=1)
	
def setup_state_tracker(config_data, for_training=True, verbose=0, tab_level=0):
	series_code = 'RXXXX1'
	tracker_name = 'drivethru_0001_smallnet_mnist'
	config_data['training']['series_name'] = 'Custom_' + str(series_code)

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