import argparse, os
from OANN.src.utils import parse_bool_from_string, strbool_description, readjust_bools


def bonn_examples(parser):

    """
    python main_BONN.py --mode BONN --data donut_example  --elasticsize 48  --show_fig_and_exit 0 --debug_toggles 00000
    python main_BONN.py --mode BONN --data donut_example  --submode eval  --debug_toggles 00000
    python main_BONN.py --mode BONN --data donut_example  --submode random_inspect 

    python main_BONN.py --mode BONN --data big_donut_example --elasticsize 256 --show_fig_and_exit 0 --debug_toggles 00000
    python main_BONN.py --mode BONN --data big_donut_example  --submode eval  --debug_toggles 00000
    """
    parser.add_argument('--N_data', default=100, type=int, help=None)
    parser.add_argument('--test_data_spread', default=0.2, type=float, help=None)
    parser.add_argument('--elasticsize', default=9, type=int, help=None)
    BOOLS = { # see strbool_description
        'show_fig_and_exit': 0,
    }
    for bkey,b in BOOLS.items():
        parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)

    print('bonn_examples...submode: %s'%(str(dargs['submode'])))
    if dargs['data'] in ['donut_example','big_donut_example']:
        if dargs['submode'] is None:
            from OANN.examples.donut_example import train_donut_example
            train_donut_example(dargs, label_mode='discrete')
        elif dargs['submode'] == 'eval':
            from OANN.examples.donut_example import eval_donut_example
            eval_donut_example(dargs, label_mode='discrete')
        elif dargs['submode'] == 'random_inspect':
            from OANN.examples.donut_example import random_inspect_donut_example
            random_inspect_donut_example(dargs, label_mode='discrete')
        else:
            raise NotImplementedError()
    elif dargs['submode'] == 'multidonut_example':
        raise NotImplementedError()
        # donut_example(dargs, label_mode='multidim')
    else:
        raise NotImplementedError()





