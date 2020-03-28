from utils.utils import *
import pipeline.data.load_data as ioload
import pipeline.data.showcase_data as show
import pipeline.data.showcase_data_cifar as show_cifar

DATA_INFO = """Available modes:
Note: submode 0 is reserved for adhoc testing 

python main.py --mode data --submode 0
python main.py --mode data --submode mnist
python main.py --mode data --submode showcase_mnist
python main.py --mode data --submode showcase_mnist_2
python main.py --mode data --submode showcase_mnist_3
python main.py --mode data --submode showcase_cifar
python main.py --mode data --submode showcase_cifar_2
python main.py --mode data --submode showcase_cifar_3
"""

def select_data_mode(config_data):
	# __config_data is NOT YET IMPLEMENTED__
	print("select_data_mode(). submode:%s. INTERNAL SWITCH? YES."%(str(config_data['console_submode'])))
	#### INTERNAL SWITCH ####
	verbose = 250

	if config_data['console_submode'] is None:
		print(DATA_INFO)
	elif str(config_data['console_submode']) == '0':
		data_test(config_data)
	elif str(config_data['console_submode']) == 'mnist':
		ioload.load_mnist_0001(config_data, verbose=verbose)
	elif str(config_data['console_submode']) == 'showcase_mnist':
		show.showcase_mnist_0001(config_data, verbose=verbose)
	elif str(config_data['console_submode']) == 'showcase_mnist_2':
		show.showcase_mnist_0002(config_data, verbose=verbose)
	elif str(config_data['console_submode']) == 'showcase_mnist_3':
		show.showcase_mnist_0003(config_data, verbose=verbose)
	elif str(config_data['console_submode']) == 'showcase_cifar':
		show_cifar.showcase_cifar10_0001(config_data, verbose=250)
	elif str(config_data['console_submode']) == 'showcase_cifar_2':
		show_cifar.showcase_cifar10_0002(config_data, verbose=250)
	elif str(config_data['console_submode']) == 'showcase_cifar_3':
		show_cifar.showcase_cifar10_0003(config_data, verbose=250)
	else:
		print(DATA_INFO)

def data_test(config_data):
	print('data_test().')
