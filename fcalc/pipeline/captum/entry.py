from utils.utils import *

CAPTUM_INFO = """Available modes:
Note: submode 0 is reserved for adhoc testing 

__NOT_IMPLEMENTED__

python main.py --mode captum --submode 0
python main.py --mode captum --submode smallnet_mnist_test
"""

def select_captum_mode(config_data):
	print('select_captum_mode. submode:%s'%(str(config_data['console_submode'])))

	if config_data['console_submode'] is None: 
		print(CAPTUM_INFO)
	elif str(config_data['console_submode']) == '0':
		captum_test(config_data)
	else:
		print(CAPTUM_INFO)


def captum_test(config_data):
	print('captum_test()')