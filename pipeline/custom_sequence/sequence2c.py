from utils.utils import *
from pipeline.custom_sequence.sequence2 import custom_sequence_abstract_0002

def custom_sequence_0002c(config_data, tab_level=0, verbose=250):
	print('custom_sequence_0002c(). MNIST. vgg')
	tracker_name = 'drivethru_0001_vgg_mnist'
	LIST_OF_LAYERS_TO_OBSERVE = ['fn'+str(i) for i in range(1, 3+1)] + ['convb_'+str(i) for i in range(1,5+1)] 
	LIST_OF_DATA_NAME = ['positive_mean_power', 'negative_mean_power', 'average_mean_power']
	
	custom_sequence_abstract_0002(config_data, tracker_name, 
		LIST_OF_LAYERS_TO_OBSERVE,LIST_OF_DATA_NAME,
		tab_level=tab_level, verbose=verbose)