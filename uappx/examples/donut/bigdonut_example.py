import os, joblib
import numpy as np
from src.utils import parse_bool_from_string, strbool_description, readjust_bools

from .donut_example import train_, eval_train_, eval_, showcase_, find_

def get_admission_th(L):
    if L==1:
        return 0.7
    else:
        return 0.5

def big_donut_example(args, dargs,parser, TOGGLES):
    print('big_donut_example...')

    parser.add_argument('--submode', default='train', type=str, help=None)
    parser.add_argument('--kwidth', default=16, type=int, help=None)
    parser.add_argument('--redir_id', default=0, type=int, help=None)
    BOOLS = { # if any
        'show_fig_and_exit': 1, 
    }
    for bkey,b in BOOLS.items():
        parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)

    if dargs['submode'] == 'train':
        extra_settings= {
            'activation_threshold': 0.9,
            'admission_threshold': get_admission_th,            
        }
        train_(dargs, extra_settings=extra_settings)
    elif dargs['submode'] == 'eval_train':
        eval_train_(dargs)
    elif dargs['submode'] == 'eval':
        eval_(dargs)
    elif dargs['submode'] == 'showcase':
        settings = {
            'subplot1':{
                'annotate':False,
            },
            'subplot2':{
                'background_alpha':0.05,
                'annotate':False,
            }
        }
        showcase_(parser, BOOLS, settings)
    elif dargs['submode'] == 'find':
        find_(parser, BOOLS)
    elif dargs['submode'] == 'showcase':
        showcase_(parser, BOOLS)
    else:
        raise NotImplementedError('invalid submode?')

