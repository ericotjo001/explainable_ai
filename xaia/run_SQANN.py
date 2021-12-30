import argparse
def parse_bool_from_string(bool_string):
    # assume bool_string is either 0 or 1 (str)
    if str(bool_string)=='1': return True
    elif str(bool_string)=='0': return False
    else: raise RuntimeError('parse_bool_from_string only accepts 0 or 1.')
strbool_description = 'bool by string 1 or 0 (avoid store_true problem)'

MODES = """
=== Welcome to run_SQANN.py. Please read README.md ===
"""

if __name__ == '__main__':
    print('SQANN')   

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--N', default=64, type=int, help=None)
    parser.add_argument('--mode', default=None, type=str, help=None)
    parser.add_argument('--submode', default=None, type=str, help=None)

    parser.add_argument('--ROOT_DIR', default=None, type=str, help=None)
    parser.add_argument('--show_fig_and_exit', default=0, type=str, help=strbool_description)
    parser.add_argument('--test_data_spread', default=0.2, type=float, help=None)

    # --test_act 1 --test_donut_data 1 --test_first_layer 1 --test_second_layer 1
    bool_vars = ['test_act','test_donut_data','test_first_layer','test_second_layer', 'test_net_allow_miss']
    for b in bool_vars:
        parser.add_argument('--%s'%(str(b)), default=0, type=str, help=strbool_description)
        
    args = vars(parser.parse_args())  # is a dictionary
    # print(args)

    for b in bool_vars:
        args[b] = parse_bool_from_string(args[b])

    from SQANN.src.tests import run_tests
    run_tests(args)

    from SQANN.src.examples import *
    from SQANN.src.data_collect import *
    if args['mode'] is None:
        print(MODES)
    elif args['mode'] == 'example1':
        if args['submode'] is None:
            run_example1(args)
        elif args['submode'] == 'collect':
            collect_example1(args)
    elif args['mode'] == 'example2':
        if args['submode'] is None:
            run_example2(args)
        elif args['submode'] == 'collect':
            collect_example2(args)
    else:
        print(MODES)