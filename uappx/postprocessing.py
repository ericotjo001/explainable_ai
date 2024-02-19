import argparse
from src.utils import parse_bool_from_string, strbool_description

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--mode', default='hyperboxplots', type=str, help=None)
    # parser.add_argument('--ROOT_DIR', default=None, type=str, help=None)
    # parser.add_argument('--PROJECT_NAME', default='srd_project', type=str, help=None)
    # parser.add_argument('--GARAM_MASALA', default=100, type=int, help=None)
    # parser.add_argument('--JALAPENO', default=100, type=int, help=None)
    # parser.add_argument('--id', nargs='+', default=['a','b']) # for list args

    # BOOLS = { # see strbool_description
    #     'BOOL': 1,
    #     'BOOL2': 0,
    # }
    # for bkey,b in BOOLS.items():
    #     parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)
    
    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary

    # print('readjust bools...')
    # args, dargs = readjust_bools(args, dargs, BOOLS)

    # # See results here
    # print(dargs)

    # TOGGLES = [parse_bool_from_string(x) for x in dargs['debug_toggles']]
    # print(TOGGLES)

    if dargs['mode'] == 'hyperboxplots':
        print('hyperboxplots!')
        from examples.entry import postprocessing
        postprocessing(dargs)
    else:
        raise NotImplementedError('mode not recognized')
