import argparse
from OANN.src.utils import parse_bool_from_string, strbool_description, readjust_bools

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)
    parser.add_argument('--mode', default='playground', type=str, help=None)
    parser.add_argument('--submode', default=None, type=str, help=None)
    parser.add_argument('--data', default=None, type=str, help=None)
    parser.add_argument('--verbose', default=100, type=int, help=None)
    # parser.add_argument('--ROOT_DIR', default=None, type=str, help=None)
    # parser.add_argument('--PROJECT_NAME', default='srd_project', type=str, help=None)
    # parser.add_argument('--id', nargs='+', default=['a','b']) # for list args

    BOOLS = { # if any
        # 'BOOL': 1, 
    }
    for bkey,b in BOOLS.items():
        parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)
    parser.add_argument('--debug_toggles', default='0000000',type=str,help='Only string of 0 and 1') 
    # TOGGLES = [parse_bool_from_string(x) for x in args['debug_toggles']]   
    
    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)

    # TOGGLES = [parse_bool_from_string(x) for x in dargs['debug_toggles']]
    # print(TOGGLES)

    if dargs['mode'] == 'playground':
        from OANN.playground import run_playground
        run_playground(parser)
    elif dargs['mode'] == 'SQANN':
        from OANN.sqann_example import sqann_examples
        sqann_examples(parser)
    elif dargs['mode'] == 'BONN':
        from OANN.bonn_example import bonn_examples
        bonn_examples(parser)
    else:
        raise NotImplementedError('Invalid mode')