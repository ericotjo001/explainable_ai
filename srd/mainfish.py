import argparse

def parse_bool_from_string(bool_string):
    # assume bool_string is either 0 or 1 (str)
    if str(bool_string)=='1': return True
    elif str(bool_string)=='0': return False
    else: raise RuntimeError('parse_bool_from_string only accepts 0 or 1.')
strbool_description = 'bool by string 1 or 0 (avoid store_true problem)'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--mode', default='fish', type=str, help=None)   
    parser.add_argument('--n_iter', default=256, type=int, help=None)
    
    # plotting
    parser.add_argument('--ROOT_DIR', default=None, type=str, help=None)
    parser.add_argument('--PROJECT_NAME', default='robotfish1D', type=str, help=None)
    parser.add_argument('--start_index', default=0, type=int, help=None)
    parser.add_argument('--end_index', default=None, type=int, help=None)
    parser.add_argument('--average_every', default=24, type=int, help=None)
    # parser.add_argument('--BOOL', default=1, type=str, help=strbool_description)
    # parser.add_argument('--BOOL2', default=0, type=str, help=strbool_description)
    # parser.add_argument('--id', nargs='+', default=['a','b']) # for list args

    ####################################
    # debug toggles
    ####################################
    ALL_BOOLS = [
        # Layer/Module design
        'test_FLD','test_FC','test_FNN', 'test_PFC', 'test_cognitive_run', 
        # map
        'test_env',
    ]
    
    for this_bool in ALL_BOOLS: 
        parser.add_argument('--%s'%(str(this_bool)), default=0, type=str, help=strbool_description)

    args = vars(parser.parse_args())  # is a dictionary
    # print(args)

    for this_bool in ALL_BOOLS:
        args[this_bool] = parse_bool_from_string(args[this_bool])
    # if parse_bool_from_string(args['BOOL']): print('true')
    # if not parse_bool_from_string(args['BOOL2']): print('also true')

    if args['mode'] == 'fish':
        from src.basic.fish import run_fish
        run_fish(args)

    elif args['mode'] == 'fish_srd':
        from src.basic.fish import run_fish_srd
        run_fish_srd(args)
