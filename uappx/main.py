import argparse
from src.utils import parse_bool_from_string, strbool_description

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--mode', default='example', type=str, help=None)
    parser.add_argument('--data', default='donut', type=str, help=None)
    parser.add_argument('--type', default='classification', type=str, help=None)
    parser.add_argument('--debug_toggles', default='0000000',type=str,help='Only string of 0 and 1') 
    
    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary

    TOGGLES = [parse_bool_from_string(x) for x in dargs['debug_toggles']]

    if dargs['mode'] == 'example':
        from examples.entry import select_example
        select_example(args, dargs, parser, TOGGLES)
    else:
        raise NotImplementedError('mode not recognized')
