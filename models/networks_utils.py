from utils.utils import *

def count_parameters(model, print_param=False, tab_level=0, verbose=0):
	if print_param:
		for param in model.parameters(): print(param)
	num_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
	num_grad = sum(p.numel() for p in model.parameters())
	pm.printvm("networks.py. count_parameters()\n    with grad: %s, with or without: %s"%(num_with_grad, num_grad), 
		tab_level=tab_level, verbose=verbose, verbose_threshold=100)
	return num_with_grad, num_grad