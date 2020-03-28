from utils.utils import *

"""
pm is printing manager from utils/printing_manager.py
"""


def get_optimizer(this_net, config_data, verbose=0, tab_level=0):
	pm.printvm("get_optimizer(). Setting up: %s"%(str(config_data['learning']['mechanism'])),
		 tab_level=tab_level, tab_shape='  ',verbose=verbose, verbose_threshold=50)
	
	if config_data['learning']['mechanism'] == 'SGD':
		conf = config_data['learning']['SGD']
		optimizer = optim.SGD(this_net.parameters(), lr=conf['learning_rate'], 
			momentum=conf['momentum'], weight_decay=conf['weight_decay'])	
	
	elif config_data['learning']['mechanism'] == 'adam':
		conf = config_data['learning']['adam']
		optimizer = optim.Adam(this_net.parameters(), lr=conf['learning_rate'],
			 betas=conf['betas'])
	
	return optimizer