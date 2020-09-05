from utils.utils import *

CUSTOM_SEQUENCE_INFO = """Available modes:
Note: submode 0 is reserved for adhoc testing 
# means not fixed yet

python main.py --mode custom_sequence --submode 0
python main.py --mode custom_sequence --submode smallnet_mnist --subsubmode RXXXX1
python main.py --mode custom_sequence --submode smallnet_cifar --subsubmode RXXXX1

python main.py --mode custom_sequence --submode alexnet_mnist --subsubmode RXXXX1
python main.py --mode custom_sequence --submode alexnet_cifar --subsubmode RXXXX1

python main.py --mode custom_sequence --submode smallnet_mnist_upsize --subsubmode RXXXX2
python main.py --mode custom_sequence --submode alexnet_mnist_upsize --subsubmode RXXXX2
python main.py --mode custom_sequence --submode smallnet_cifar_upsize --subsubmode RXXXX2
python main.py --mode custom_sequence --submode alexnet_cifar_upsize --subsubmode RXXXX2


python main.py --mode custom_sequence --submode alexnet_mnist --subsubmode RXXXX1
python main.py --mode custom_sequence --submode alexnet_mnist_upsize --subsubmode RXXXX2
"""
def select_custom_sequence_mode(config_data):
	print('select_custom_sequence_mode(). submode:%s. HARDCODE VAR? YES.'%(str(config_data['console_submode'])))
	if config_data['console_submode'] is None: 
		print(CUSTOM_SEQUENCE_INFO)
	elif str(config_data['console_submode']) == '0':
		custom_test(config_data)
	elif str(config_data['console_submode']) == 'smallnet_mnist':
		import pipeline.custom_sequence.seq_smallnet_mnist as this_seq
		this_seq.seq_smallnet_mnist(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'smallnet_mnist_upsize':
		import pipeline.custom_sequence.seq_smallnet_mnist as this_seq
		this_seq.seq_smallnet_mnist_upsize(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_mnist':
		import pipeline.custom_sequence.seq_alexnet_mnist as this_seq
		this_seq.seq_alexnet_mnist(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_mnist_upsize':
		import pipeline.custom_sequence.seq_alexnet_mnist as this_seq
		this_seq.seq_alexnet_mnist_upsize(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'smallnet_cifar':
		import pipeline.custom_sequence.seq_smallnet_cifar as this_seq
		this_seq.seq_smallnet_cifar(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'smallnet_cifar_upsize':
		import pipeline.custom_sequence.seq_smallnet_cifar as this_seq
		this_seq.seq_smallnet_cifar_upsize(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_cifar':
		import pipeline.custom_sequence.seq_alexnet_cifar as this_seq
		this_seq.seq_alexnet_cifar(config_data, tab_level=0, verbose=250)
	elif str(config_data['console_submode']) == 'alexnet_cifar_upsize':
		import pipeline.custom_sequence.seq_alexnet_cifar as this_seq
		this_seq.seq_alexnet_cifar_upsize(config_data, tab_level=0, verbose=250)
	else:
		print(CUSTOM_SEQUENCE_INFO)
		
def custom_test(config_data):
	print('custom_test()')