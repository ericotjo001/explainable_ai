from utils.utils import *
from pipeline.data.load_data_cifar import load_cifar_0001
from pipeline.evaluation.eval_smallnet_mnist_debug import DEBUG_eval_smallnet_mnist_LOOP_0001
from pipeline.evaluation.eval_smallnet_cifar import eval_cifar_0001
import pipeline.training.train_alexnet_cifar as tr

if IS_DEBUG:
	DEBUG_eval_alexnet_cifar_LOOP_SIGNAL = 100
else:
	DEBUG_eval_alexnet_cifar_LOOP_SIGNAL = 0

def eval_alexnet_cifar_0002_test(config_data, tab_level=0, verbose=250):
	print('eval_alexnet_cifar_0002_test()')
	state_tracker = tr.setup_state_tracker(config_data, for_training=False, verbose=verbose, tab_level=tab_level)
	net = tr.new_or_load_model(state_tracker, config_data, verbose=verbose, tab_level=tab_level)
	eval_cifar_0001(config_data, state_tracker, net, DEBUG_eval_alexnet_cifar_LOOP_SIGNAL, train=False, tab_level=0, verbose=250)