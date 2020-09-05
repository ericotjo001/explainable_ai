from utils.utils import *

if IS_DEBUG:
	DEBUG_train_smallnet_mnist_LOOP_SIGNAL = 0
	DEBUG_train_smallnet_mnist_LOOP2_SIGNAL = 1
else:
	DEBUG_train_smallnet_mnist_LOOP_SIGNAL = 0
	DEBUG_train_smallnet_mnist_LOOP2_SIGNAL = 0

"""
pm is printing manager. Refer to utils/printing_manager.py
"""

def DEBUG_train_loop_0001(DEBUG_train_smallnet_mnist_LOOP_SIGNAL, net, x, y0, tab_level=0, verbose=0):
	"""
	e.g.
	train_smallnet_mnist_debug.py. DEBUG_train_loop_0001().
	  x.shape:torch.Size([4, 1, 28, 28]),y:tensor([0, 2, 8, 3])
	  SmallCNN(). forward_debug()
	    [1] x.shape:torch.Size([4, 24, 28, 28])
	    [2] x.shape:torch.Size([4, 24, 28, 28])
	    [2.1] x.shape:torch.Size([4, 24, 28, 28])
	    [3] x.shape:torch.Size([4, 10, 1, 1])
	    [Output] x.shape:torch.Size([4, 10])
	"""
	DEBUG_SIGNAL = 0
	if DEBUG_train_smallnet_mnist_LOOP_SIGNAL:
		pm.printvm('train_smallnet_mnist_debug.py. DEBUG_train_loop_0001().'%(),
			tab_level=tab_level, tab_shape='  ',verbose=0, verbose_threshold=None)
		pm.printvm('x.shape:%s,y0:%s'%(str(x.shape),str(y0)), 
			tab_level=tab_level+1, tab_shape='  ',verbose=0, verbose_threshold=None)
		net.forward_debug(x, verbose=verbose, tab_level=tab_level+1)
		DEBUG_SIGNAL = True
	return DEBUG_SIGNAL


def DEBUG_train_loop_0002(DEBUG_train_smallnet_mnist_LOOP2_SIGNAL, n_iter, n_epoch ,tab_level=0, verbose=0):
	DEBUG_SIGNAL = [0,0]
	# n_epoch n_iter are both zero based
	if DEBUG_train_smallnet_mnist_LOOP2_SIGNAL:
		N_ITER_MAX = 10
		N_EPOCH_MAX = 2
		PRINT_EVERY_N_EPOCH = 1
		if n_iter >= N_ITER_MAX-1:
			DEBUG_SIGNAL[0] = 1
			if (n_epoch+1)%PRINT_EVERY_N_EPOCH==0:
				pm.printvm('train_smallnet_mnist_debug.py. DEBUG_train_loop_0002(). MAX_iter_REACHED. n_epoch:%s n_iter:%s'%(
					str(n_epoch), str(n_iter+1)), tab_level=tab_level+1, tab_shape='  ',verbose=0, verbose_threshold=None)		
		if n_epoch > N_EPOCH_MAX:
			DEBUG_SIGNAL[1] = 1
			if n_iter == 0:
				pm.printvm('train_smallnet_mnist_debug.py. DEBUG_train_loop_0002(). Now at MAX_EPOCH. n_epoch:%s'%(str(n_epoch)),
					tab_level=tab_level, tab_shape='  ',verbose=0, verbose_threshold=None)
	return DEBUG_SIGNAL