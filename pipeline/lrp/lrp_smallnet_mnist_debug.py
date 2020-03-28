from utils.utils import *

if IS_DEBUG:
	DEBUG_LOOP1_SIGNAL = 0
	DEBUG_LOOP2_SIGNAL = 10
else:
	DEBUG_LOOP1_SIGNAL = False
	DEBUG_LOOP2_SIGNAL = 0

def DEBUG_LOOP1(i,x,y,y0,net, DEBUG_LOOP1_SIGNAL,tab_level=0):
	DEBUG_SIGNAL = 0
	if DEBUG_LOOP1_SIGNAL:
		DEBUG_SIGNAL = True

		pm.printvm('[%s] DEBUG_LOOP1()'%(str(i)
			),tab_level=tab_level,verbose=0, verbose_threshold=None)	
		R = net.relprop_debug(y, verbose=250, tab_level=tab_level)		

	return DEBUG_SIGNAL

def DEBUG_LOOP2(i,x,y,y0,R, net, DEBUG_LOOP2_SIGNAL,tab_level=0):
	DEBUG_SIGNAL = 0

	if DEBUG_LOOP2_SIGNAL > 0:
		if i>= DEBUG_LOOP2_SIGNAL:
			DEBUG_SIGNAL=True
		else:
			pm.printvm('[%s] R.shape:%s'%(str(i),str(R.shape),
				),tab_level=tab_level,verbose=0, verbose_threshold=None)			

	return DEBUG_SIGNAL