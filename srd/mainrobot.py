import argparse, os

def parse_bool_from_string(bool_string):
    # assume bool_string is either 0 or 1 (str)
    if str(bool_string)=='1': return True
    elif str(bool_string)=='0': return False
    else: raise RuntimeError('parse_bool_from_string only accepts 0 or 1.')
strbool_description = 'bool by string 1 or 0 (avoid store_true problem)'

from src.robot2d.utils import only_odd_map_size
from src.robot2d.robot import run_robot, run_robot_srd, run_robot_eval, run_robot_srd_training, aggregate_result
from src.robot2d.robot_lava import run_robot_eval_lava, run_robot_srd_training_lava

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--ROOT_DIR', default=None, type=str, help=None)  
    parser.add_argument('--PROJECT_NAME', default=None, type=str, help=None)  
    parser.add_argument('--CHECKPOINT_LAYER', default=None, type=str, help=None)  

    parser.add_argument('--mode', default='robot', type=str, help=None)  
    parser.add_argument('--map_size', nargs='+', default=[11,13], type=int) # for list args
    parser.add_argument('--iter_limit', default=36, type=int, help=None) # must reach target within this iters
    parser.add_argument('--n_maps', default=8, type=int, help=None) 
    parser.add_argument('--n_plans', default=2, type=int, help=None) 
    
    parser.add_argument('--map_data_name', default=None, type=str, help=None)  
    parser.add_argument('--train_data_name', default=None, type=str, help=None)  
    parser.add_argument('--eval_after_train', default=1, type=str, help=strbool_description)   

    parser.add_argument('--n_expt', default=None, type=int, help=None) 
    parser.add_argument('--custom_tile_val', default=0, type=str, help=strbool_description) 
    parser.add_argument('--custom_target', default=10.0, type=float, help=None) 
    parser.add_argument('--custom_dirt', default=0.2, type=float, help=None) 
    parser.add_argument('--custom_grass', default=-0.8, type=float, help=None) 

    parser.add_argument('--lava_fraction', default=0., type=float, help=None)
    parser.add_argument('--include_lava', default=0, type=str, help=strbool_description)  
    parser.add_argument('--unknown_avoidance', default=2., type=float, help=None)

    ####################################
    # debug toggles
    ####################################
    ALL_TEST_BOOLS = [
         # map
        'test_map', 'peek_map',
        # module/layer
        'test_act','test_act2', 'test_local_act', 'test_plan', 'test_srd',

    ]
    for this_bool in ALL_TEST_BOOLS: 
        parser.add_argument('--%s'%(str(this_bool)), default=0, type=str, help=strbool_description)

    args = vars(parser.parse_args())  # is a dictionary
    # print(args)
    for this_bool in ALL_TEST_BOOLS:
        args[this_bool] = parse_bool_from_string(args[this_bool])
    args['ALL_TEST_BOOLS'] = ALL_TEST_BOOLS

    OTHER_BOOLS = ['eval_after_train','custom_tile_val','include_lava']
    for this_bool in OTHER_BOOLS:
        args[this_bool] = parse_bool_from_string(args[this_bool])
    only_odd_map_size(args)

    #####################################
    # To run and save robot exploration images
    #####################################
    if args['mode'] == 'robot': 
        run_robot(args) 
    elif args['mode'] == 'robot_srd':
        run_robot_srd(args)
    elif args['mode'] == 'data':
        from src.robot2d.data import store_map_data
        store_map_data(args)

    #####################################
    # For experimental data collection
    #####################################

    elif args['mode'] == 'robot_eval':        
        run_robot_eval(args, run_srd=False)
    elif args['mode'] == 'robot_srd_train':        
        run_robot_srd_training(args)
    elif args['mode'] == 'robot_srd_eval':
        run_robot_eval(args, run_srd=True)

    elif args['mode'] == 'run_through':
        assert(args['map_data_name'] is not None)
        assert(args['train_data_name'] is not None)
        assert(args['n_expt'] is not None)

        args['CHECKPOINT_LAYER'] = args['PROJECT_NAME']  
        PROJECT_HEADER = args['PROJECT_NAME'] 

        for i in range(args['n_expt']):
            print('start expt %s'%(str(i)))
            args['PROJECT_NAME'] = PROJECT_HEADER + str(1001+i)[1:]
            run_robot_eval(args, run_srd=False)
            args['PROJECT_NAME'] = args['PROJECT_NAME'] + '.srd'
            run_robot_srd_training(args)
            print()


    elif args['mode'] == 'run_through_lava':
        assert(args['map_data_name'] is not None)
        assert(args['train_data_name'] is not None)
        assert(args['n_expt'] is not None)

        args['CHECKPOINT_LAYER'] = args['PROJECT_NAME']  
        PROJECT_HEADER = args['PROJECT_NAME'] 

        for i in range(args['n_expt']):
            print('start expt %s'%(str(i)))
            args['PROJECT_NAME'] = PROJECT_HEADER + str(1001+i)[1:]
            run_robot_eval_lava(args, run_srd=False)
            args['PROJECT_NAME'] = args['PROJECT_NAME'] + '.srd'
            run_robot_srd_training_lava(args)
            print()


    elif args['mode'] == 'aggregate_result':
        assert(args['n_expt'] is not None)
        aggregate_result(args)

    elif args['mode'] == 'compare_weights':
        from src.robot2d.visualize_weights import compare_weights
        compare_weights(args)