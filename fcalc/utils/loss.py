from utils.utils import *

def compute_loss(criterion, y, y0):
	loss = criterion(y, y0)
	
	"""
	Add custom loss here
	"""
	
	return loss