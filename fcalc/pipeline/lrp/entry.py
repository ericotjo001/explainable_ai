from utils.utils import *

LRP_INFO = """Available modes:
Note: submode 0 is reserved for adhoc testing 
# means not fixed yet

python main.py --mode lrp --submode 0
# python main.py --mode lrp --submode smallnet_mnist 
# python main.py --mode lrp --submode alexnet_mnist 
# python main.py --mode lrp --submode vgg_mnist 
"""

def select_lrp_mode(config_data):
	print('select_lrp_mode(). submode:%s. HARDCODE VAR? YES.'%(str(config_data['console_submode'])))
	if config_data['console_submode'] is None: 
		print(LRP_INFO)
	elif str(config_data['console_submode']) == '0':
		lrp_test(config_data)
	elif str(config_data['console_submode']) == 'smallnet_mnist':
		import pipeline.lrp.lrp_smallnet_mnist as lrp
		lrp.lrp_smallnet_mnist_0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_mnist':
		import pipeline.lrp.lrp_alexnet_mnist as lrp
		lrp.lrp_alexnet_mnist_0001(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'vgg_mnist':
		import pipeline.lrp.lrp_vgg_mnist as lrp
		lrp.lrp_vgg_mnist_0001(config_data, tab_level=0, verbose=250)
	else:
		print('\n** Invalid submode selected!\n')
		print(LRP_INFO)
		
def lrp_test(config_data):
	print('lrp_test()')