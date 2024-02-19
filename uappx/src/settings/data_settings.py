import os

MNIST_SETTING = {
    'SOURCE_DATA_DIR': 'C:/data', # downloaded from the torchvision
    'DATA_DIR': os.path.join('data', 'MNIST', 'train'), # reshaped into the desired folder
    'TEST_DATA_DIR': os.path.join('data', 'MNIST', 'test'), # reshaped into the desired folder
}

CIFAR_SETTING = {
    'SOURCE_DATA_DIR': 'C:/data/cifar10', 
    'DATA_DIR': os.path.join('data', 'CIFAR', 'train'), # reshaped into the desired folder
    'TEST_DATA_DIR': os.path.join('data', 'CIFAR', 'test'), # reshaped into the desired folder    
}

IMAGENET_SETTING = {
    'SOURCE_DATA_DIR': 'C:/data/ImageNet/ILSVRC/Data/CLS-LOC',
}

IMAGENETV2_SETTING = {
    'MFREQ_DATA_DIR': 'C:/data/ImageNetV2/imagenetv2-matched-frequency-format-val',
    '0.7FREQ_DATA_DIR': 'C:/data/ImageNetV2/imagenetv2-threshold0.7-format-val',
}