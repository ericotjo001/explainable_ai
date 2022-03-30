import argparse

from OANN.src.data import DonutDataX
import OANN.src.model as model
from OANN.src.utils import parse_bool_from_string, strbool_description, readjust_bools,\
    simple_evaluation, standard_evaluation

def sqann_examples(parser):
    """    
    python main_BONN.py --mode SQANN --submode donut_example --show_fig_and_exit 0
    python main_BONN.py --mode SQANN --submode multidonut_example --show_fig_and_exit 0
    python main_BONN.py --mode SQANN --submode discretedonut_example --show_fig_and_exit 0
    """

    parser.add_argument('--N_data', default=100, type=int, help=None)
    parser.add_argument('--test_data_spread', default=0.2, type=float, help=None)
    BOOLS = { # see strbool_description
        'show_fig_and_exit': 0,
    }
    for bkey,b in BOOLS.items():
        parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)

    print('sqann_examples...submode: %s\n'%(str(dargs['submode'])))
    if dargs['submode'] == 'donut_example':
        donut_example(dargs, label_mode='scalar')
    elif dargs['submode'] == 'multidonut_example':
        donut_example(dargs, label_mode='multidim')
    elif dargs['submode'] == 'discretedonut_example':
        donut_example(dargs, label_mode='discrete')

def donut_example(dargs, label_mode):
    print('donut_example\n\n')

    ld = DonutDataX(dargs['N_data'], test_sd=dargs['test_data_spread'], label_mode=label_mode,
        show_fig_and_exit=dargs['show_fig_and_exit'])
    X,Y = ld.X, ld.Y
    X_test, Y_test = ld.X_test, ld.Y_test
    
    if label_mode in ['scalar', 'multidim']:
        settings = {'init_new':True, 'output_mode':'continuous' }
    elif label_mode=='discrete':
        settings = {'init_new':True, 'output_mode':'discrete', 'n_class': ld.n_class}

    net = model.SQANN(**settings)
    net.fit_data(X,Y,verbose=20)

    print()
    simple_evaluation(X,Y,net, header_text='Show fitting on training data', 
        output_mode=settings['output_mode'], verbose=100)

    print('\nevaluate on test dataset.')
    standard_evaluation(X_test, Y_test, net, output_mode=settings['output_mode'],)