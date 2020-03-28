from utils.utils import *
import pipeline.training.train_vgg_mnist as tr
import pipeline.evaluation.eval_smallnet_mnist as emn

if IS_DEBUG:
	DEBUG_LOOP_SIGNAL = 100

else:
	DEBUG_LOOP_SIGNAL = 0

def eval_vgg_mnist_0001_overfitting(config_data, tab_level=0, verbose=250):
	print('eval_vgg_mnist_0001_overfitting()')
	config_data['data_from_torch']['mnist']['training_mode'] = True
	eval_vgg_mnist_0001(config_data, tab_level=0, verbose=0)

def eval_vgg_mnist_0002_test(config_data, tab_level=0, verbose=250):
	print('eval_vgg_mnist_0002_test().')
	config_data['data_from_torch']['mnist']['training_mode'] = False
	eval_vgg_mnist_0001(config_data, tab_level=0, verbose=0)

def eval_vgg_mnist_0001(config_data, tab_level=0, verbose=0):
	state_tracker = tr.setup_state_tracker(config_data, for_training=False, verbose=verbose, tab_level=tab_level)
	net = tr.new_or_load_model(state_tracker, verbose=verbose, tab_level=tab_level)
	# this function is called from eval_smallnet_mnist only because it was developed first.
	emn.eval_mnist_0001(config_data, state_tracker, net, DEBUG_LOOP_SIGNAL, 
		tab_level=tab_level, verbose=verbose)