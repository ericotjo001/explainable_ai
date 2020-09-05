from utils.utils import *

TEQ_INFO = """Available modes:
Run them in sequence:

python main.py --mode training --submode smallnet_mnist
python main.py --mode evaluation --submode smallnet_mnist_overfit
python main.py --mode evaluation --submode smallnet_mnist_test


python main.py --mode training --submode smallnet_cifar
python main.py --mode evaluation --submode smallnet_cifar_test


python main.py --mode training --submode alexnet_mnist
python main.py --mode evaluation --submode alexnet_mnist_overfit
python main.py --mode evaluation --submode alexnet_mnist_test


python main.py --mode training --submode alexnet_cifar
python main.py --mode evaluation --submode alexnet_cifar_test


python main.py --mode drivethru --submode smallnet_mnist
python main.py --mode drivethru --submode alexnet_mnist

python main.py --mode custom_sequence --submode smallnet_mnist --subsubmode RXXXX1
python main.py --mode custom_sequence --submode smallnet_mnist_upsize --subsubmode RXXXX2
python main.py --mode custom_sequence --submode alexnet_mnist --subsubmode RXXXX1
python main.py --mode custom_sequence --submode alexnet_mnist_upsize --subsubmode RXXXX2
python main.py --mode custom_sequence --submode smallnet_cifar --subsubmode RXXXX1
python main.py --mode custom_sequence --submode smallnet_cifar_upsize --subsubmode RXXXX2
python main.py --mode custom_sequence --submode alexnet_cifar --subsubmode RXXXX1
python main.py --mode custom_sequence --submode alexnet_cifar_upsize --subsubmode RXXXX2


"""

def select_teq_mode(config_data):
	print('select_teq_mode()')
	print(TEQ_INFO)

