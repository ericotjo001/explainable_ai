from utils.utils import *

TRAINING_INFO = """Available modes:
Note: submode 0 is reserved for adhoc testing 
# means not fixed yet

python main.py --mode training --submode 0
python main.py --mode training --submode smallnet_mnist
python main.py --mode training --submode smallnet_cifar
python main.py --mode training --submode alexnet_mnist
python main.py --mode training --submode alexnet_cifar
# python main.py --mode training --submode vgg_mnist
# python main.py --mode training --submode vgg_cifar
"""

def select_training_mode(config_data):
	print('select_training_mode(). submode:%s. HARDCODE VAR? YES.'%(str(config_data['console_submode'])))
	if config_data['console_submode'] is None: 
		print(TRAINING_INFO)
	elif str(config_data['console_submode']) == '0':
		training_test(config_data)
	elif str(config_data['console_submode']) == 'smallnet_mnist':
		import pipeline.training.train_smallnet_mnist as tr
		tr.train_smallnet_mnist_0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'smallnet_cifar':
		import pipeline.training.train_smallnet_cifar as tr
		tr.train_smallnet_cifar_0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_mnist':
		import pipeline.training.train_alexnet_mnist as tr
		tr.train_alexnet_mnist_0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_cifar':
		import pipeline.training.train_alexnet_cifar as tr
		tr.train_alexnet_cifar_0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'vgg_mnist':
		import pipeline.training.train_vgg_mnist as tr
		tr.train_vgg_mnist_0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'vgg_cifar':
		import pipeline.training.train_vgg_cifar as tr
		tr.train_vgg_cifar_0001(config_data, tab_level=0, verbose=250)
	else:
		print("Invalid submode selected!")
		print(TRAINING_INFO)
		
def training_test(config_data):
	TESTING_MODE = 0

	print('training_test().testing mode:%s. HARDCODED VARIABLES? YES'%(str(TESTING_MODE)))

	if TESTING_MODE == 0:
		from pipeline.training.train_testing_unit import test_tracker
		test_tracker(config_data)
