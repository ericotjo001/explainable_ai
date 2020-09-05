from utils.utils import *


def get_model_folder_path(config_data, state_tracker, tab_level=0):
	now = datetime.datetime.now()
	dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

	model_folder_name = state_tracker.training_series_name
	filename = 'log_'+ dt_string +'.txt'
	full_path_log_file = os.path.join(config_data['working_dir'],'checkpoint',model_folder_name, filename)
	pm.printvm(full_path_log_file, tab_level=tab_level)
	return full_path_log_file

def print_partition():
	print()
	print("="*80)
	print()