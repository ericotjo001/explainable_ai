from utils.descriptions.main_description import NULL_DESCRIPTION
from utils.descriptions.data_description import DATA_DESCRIPTION


def select_data_mode(console_modes):
	print('selecting data mode...')

	if console_modes['mode2'] is None:
		print(DATA_DESCRIPTION)
	elif console_modes['mode2'] == 'ten_classes':
		from .prepare_10classes_data import generate_ten_classes_training_val_test_data
		generate_ten_classes_training_val_test_data()