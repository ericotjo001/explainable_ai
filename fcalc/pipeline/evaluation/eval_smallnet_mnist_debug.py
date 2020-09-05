from utils.utils import *

if IS_DEBUG:
	DEBUG_eval_smallnet_mnist_LOOP_SIGNAL = 100

else:
	DEBUG_eval_smallnet_mnist_LOOP_SIGNAL = 0

def DEBUG_eval_smallnet_mnist_LOOP_0001(i, x, y0, y, net, DEBUG_eval_smallnet_mnist_LOOP_SIGNAL,tab_level=0):
	DEBUG_SIGNAL = 0
	PRINT_EVERY = 10
	COMPACT_PRINT = 1
	if DEBUG_eval_smallnet_mnist_LOOP_SIGNAL>0:
		if i >= DEBUG_eval_smallnet_mnist_LOOP_SIGNAL: 
			DEBUG_SIGNAL = True
		else:
			if (i+1)%PRINT_EVERY == 0:
				y1 = torch.argmax(y)
				DEBUG_AUX_PRINTING(i, COMPACT_PRINT, PRINT_EVERY,x, y, y0, y1, tab_level)
	return DEBUG_SIGNAL

def DEBUG_AUX_PRINTING(i, COMPACT_PRINT, PRINT_EVERY, x, y, y0, y1, tab_level):
	if not COMPACT_PRINT:
		pm.printvm('[%s] x.shape:%s'%(str(i), str(x.detach().cpu().numpy().shape),
			),tab_level=tab_level,verbose=0, verbose_threshold=None)
		pm.printvm('y:%s\ny0:%s\ny1:<%s>'%(
			str(y.detach().cpu().numpy()),
			str(y0.detach().cpu().numpy()),
			str(y1.detach().cpu().numpy()),
			),tab_level=tab_level+1,verbose=0, verbose_threshold=None)
	else:
		if PRINT_EVERY == i+1:
			pm.printvm('%4s | %-16s | %-5s | %-5s | %s'%('','x.shape','y0','y1','y.shape')
				,tab_level=tab_level,verbose=0, verbose_threshold=None)

		pm.printvm('%4s | %16s | %5s | %5s | %s'%(str(i), 
			str(x.detach().cpu().numpy().shape),
			str(y0.detach().cpu().numpy()),
			str(y1.detach().cpu().numpy()),
			str(y.detach().cpu().numpy().shape)
			),tab_level=tab_level,verbose=0, verbose_threshold=None)