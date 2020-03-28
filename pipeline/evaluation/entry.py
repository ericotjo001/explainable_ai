from utils.utils import *

EVAL_INFO = """Available modes:
Note: submode 0 is reserved for adhoc testing 
# means not fixed yet

python main.py --mode evaluation --submode 0
python main.py --mode evaluation --submode smallnet_mnist_overfit
python main.py --mode evaluation --submode smallnet_mnist_test
python main.py --mode evaluation --submode smallnet_cifar_test
python main.py --mode evaluation --submode alexnet_mnist_overfit
python main.py --mode evaluation --submode alexnet_mnist_test
python main.py --mode evaluation --submode alexnet_cifar_test
# python main.py --mode evaluation --submode vgg_mnist_overfit
# python main.py --mode evaluation --submode vgg_mnist_test
# python main.py --mode evaluation --submode vgg_cifar_test
"""

def select_evaluation_mode(config_data):
	print('select_evaluation_mode(). submode:%s. HARDCODE VAR? YES.'%(str(config_data['console_submode'])))
	if config_data['console_submode'] is None: 
		print(EVAL_INFO)
	elif str(config_data['console_submode']) == '0':
		evaluation_test(config_data)
	elif str(config_data['console_submode']) == 'smallnet_mnist_overfit':
		import pipeline.evaluation.eval_smallnet_mnist as ev
		ev.eval_smallnet_mnist_0001_overfitting(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'smallnet_mnist_test':
		import pipeline.evaluation.eval_smallnet_mnist as ev
		ev.eval_smallnet_mnist_0002_test(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'smallnet_cifar_test':
		import pipeline.evaluation.eval_smallnet_cifar as ev
		ev.eval_smallnet_cifar_0002_test(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_mnist_overfit':
		import pipeline.evaluation.eval_alexnet_mnist as ev
		ev.eval_alexnet_mnist_0001_overfitting(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_mnist_test':
		import pipeline.evaluation.eval_alexnet_mnist as ev
		ev.eval_alexnet_mnist_0002_test(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_cifar_test':
		import pipeline.evaluation.eval_alexnet_cifar as ev
		ev.eval_alexnet_cifar_0002_test(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'vgg_mnist_overfit':
		import pipeline.evaluation.eval_vgg_mnist as ev
		ev.eval_vgg_mnist_0001_overfitting(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'vgg_mnist_test':
		import pipeline.evaluation.eval_vgg_mnist as ev
		ev.eval_vgg_mnist_0002_test(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'vgg_cifar_test':
		import pipeline.evaluation.eval_vgg_cifar as ev
		ev.eval_vgg_cifar_0002_test(config_data, tab_level=0, verbose=250)
	else:
		print(EVAL_INFO)

def evaluation_test(config_data):
	print('evaluation_test()')