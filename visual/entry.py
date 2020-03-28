from utils.utils import *
import visual.visual_torch_data as to
import visual.visual_small_net_mnist as visnm

VISUAL_INFO = """Available modes:
Note: submode 0 is reserved for adhoc testing 

python main.py --mode visual --submode 0
python main.py --mode visual --submode data0001
python main.py --mode visual --submode smallnet_mnist_visual_final_LRP_output
python main.py --mode visual --submode alexnet_mnist_visual_final_LRP_output
python main.py --mode visual --submode drivethru_smallnet_mnist
python main.py --mode visual --submode drivethru_smallnet_cifar
python main.py --mode visual --submode drivethru_alexnet_mnist
python main.py --mode visual --submode drivethru_alexnet_cifar
python main.py --mode visual --submode drivethru_vgg_mnist
python main.py --mode visual --submode drivethru_vgg_cifar
"""

def select_visual_mode(config_data):
	print('select_visual_mode()')

	if config_data['console_submode'] is None:
		print(VISUAL_INFO)
	elif str(config_data['console_submode']) == '0':
		visual_test(config_data)
	elif str(config_data['console_submode']) == 'data0001':
		to.visual_mnist(config_data)
	elif str(config_data['console_submode']) == 'smallnet_mnist_visual_final_LRP_output':
		visnm.visual_final_LRP_output(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_mnist_visual_final_LRP_output':
		from visual.visual_alexnet_mnist import visual_final_LRP_output
		visual_final_LRP_output(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'drivethru_smallnet_mnist':
		from visual.drivethru_visual_smallnet_mnist import drivethru_visual0001
		drivethru_visual0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'drivethru_smallnet_cifar':
		from visual.drivethru_visual_smallnet_cifar import drivethru_visual0001
		drivethru_visual0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'drivethru_alexnet_mnist':
		from visual.drivethru_visual_alexnet_mnist import drivethru_visual0001
		drivethru_visual0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'drivethru_alexnet_cifar':
		from visual.drivethru_visual_alexnet_cifar import drivethru_visual0001
		drivethru_visual0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'drivethru_vgg_mnist':
		from visual.drivethru_visual_vgg_mnist import drivethru_visual0001
		drivethru_visual0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'drivethru_vgg_cifar':
		from visual.drivethru_visual_vgg_cifar import drivethru_visual0001
		drivethru_visual0001(config_data, tab_level=0, verbose=250)
	else:
		print(VISUAL_INFO)

def visual_test(config_data):
	print('visual_test()')
