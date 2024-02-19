import torch
import os, joblib
import numpy as np
from PIL import Image
from src.utils import parse_bool_from_string, strbool_description, readjust_bools
from .imagenet_prep import prep_data_and_dirs, get_admission_th, prep_deep_neural_network_and_data_loader
from .dnn import device


def run_imagenet_example(args, dargs, parser):
    print('run_imagenet_example...')

    parser.add_argument('--submode', default='train', type=str, help=None)
    parser.add_argument('--kwidth', default=16, type=int, help=None)

    BOOLS = { # if any
        'DNN_TRAINING':0,
    }
    for bkey,b in BOOLS.items():
        parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)

    if dargs['submode'] == 'ces':
        # cse: compare with euclidean dist, selected samples
        ces_(dargs, parser, BOOLS)
    else:
        raise NotImplementedError('invalid submode?')



def ces_(dargs, parser, BOOLS):
    from .imagenet_showcase_select import ces
    ces(dargs, parser, BOOLS)
