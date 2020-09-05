from utils.utils import *

DRIVETHRU_INFO = """Available modes:
Note: submode 0 is reserved for adhoc testing 
# means not fixed yet

python main.py --mode drivethru --submode 0
python main.py --mode drivethru --submode smallnet_mnist
python main.py --mode drivethru --submode smallnet_cifar
python main.py --mode drivethru --submode alexnet_mnist
python main.py --mode drivethru --submode alexnet_cifar
# python main.py --mode drivethru --submode vgg_mnist
# python main.py --mode drivethru --submode vgg_cifar
"""

def select_drivethru_mode(config_data):
	print('select_drivethru_mode(). submode:%s. HARDCODE VAR? YES.'%(str(config_data['console_submode'])))
	if config_data['console_submode'] is None: 
		print(DRIVETHRU_INFO)
	elif str(config_data['console_submode']) == '0':
		drivethru_test(config_data)
	elif str(config_data['console_submode']) == 'smallnet_mnist':
		import pipeline.drivethru.drivethru_smallnet_mnist as dtm
		dtm.drivethru_0001_smallnet_mnist(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'smallnet_cifar':
		import pipeline.drivethru.drivethru_smallnet_cifar as dtc
		dtc.drivethru_0001_smallnet_cifar(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_mnist':
		import pipeline.drivethru.drivethru_alexnet_mnist as dtm
		dtm.drivethru_0001_alexnet_mnist(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_cifar':
		import pipeline.drivethru.drivethru_alexnet_cifar as dtc
		dtc.drivethru_0001_alexnet_cifar(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'vgg_mnist':
		import pipeline.drivethru.drivethru_vgg_mnist as dtm
		dtm.drivethru_0001_vgg_mnist(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'vgg_cifar':
		import pipeline.drivethru.drivethru_vgg_cifar as dtc
		dtc.drivethru_0001_vgg_cifar(config_data, tab_level=0, verbose=250)
	else:
		print('\n** Invalid submode selected!\n')
		print(DRIVETHRU_INFO)

def drivethru_test(config_data):
	print("drivethru_test()")