from utils.utils import *
from pipeline.evaluation.eval_smallnet_mnist_debug import *
import pipeline.training.train_smallnet_mnist as tr
from pipeline.data.load_data import load_mnist_0001

def eval_smallnet_mnist_0001_overfitting(config_data, tab_level=0, verbose=0):
	print('eval_smallnet_mnist_0001_overfitting().')
	config_data['data_from_torch']['mnist']['training_mode'] = True
	eval_smallnet_mnist_0001(config_data, tab_level=tab_level, verbose=verbose)
	
def eval_smallnet_mnist_0002_test(config_data, tab_level=0, verbose=0):
	print('eval_smallnet_mnist_0001_test().')
	config_data['data_from_torch']['mnist']['training_mode'] = False
	eval_smallnet_mnist_0001(config_data, tab_level=tab_level, verbose=verbose)
	
def eval_smallnet_mnist_0001(config_data, tab_level=0, verbose=0):
	state_tracker = tr.setup_state_tracker(config_data, for_training=False, verbose=verbose, tab_level=tab_level)
	net = tr.new_or_load_model(state_tracker, config_data, verbose=verbose, tab_level=tab_level)
	eval_mnist_0001(config_data, state_tracker, net, DEBUG_eval_smallnet_mnist_LOOP_SIGNAL, 
		tab_level=tab_level, verbose=verbose)

def eval_mnist_0001(config_data, state_tracker, net, debug_signal, tab_level=0, verbose=250):
	from utils.metrics import ClassAccuracy
	data_loader = load_mnist_0001(config_data, shuffle=False, batch_size=1, verbose=0)	
	net.eval()
	Acc = ClassAccuracy()
	pm.printvm('Start evaluation...',
		tab_level=tab_level,verbose=verbose, verbose_threshold=250)
	for i, data in enumerate(data_loader,0):
		x, y0 = data
		y = net(x.to(this_device)).squeeze(3).squeeze(2)
		y1 = torch.argmax(y.squeeze(1)).clone().detach().cpu().numpy()
		Acc.update_acc(int(y1),int(y0))

		if DEBUG_eval_smallnet_mnist_LOOP_0001(i, x, y0, y, net, debug_signal,
			tab_level=tab_level+1): break
	Acc.compute_acc()
	Acc.display_stats(tab_level=0, verbose=250)
	"""
	Acc().display_stats()
	  N : 101
	  n_correct_pred : 7
	  acc : 0.06930693069306931

	"""