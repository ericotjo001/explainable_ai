import argparse, os
from utils import *

available_methods = 'Saliency,InputXGradient, LayerGradCam, Deconvolution, GuidedBackprop, DeepLift'
if __name__=='__main__':

    DIR_HELP = """ If run from gpt folder, set to None.\n
    If run from another folder, use:\n
    1. --ROOT_DIR_MODE abs, if you supply absolute dir to --ROOT_DIR
    2. --ROOT_DIR_MODE rel, if you supply relative dir to --ROOT_DIR
    """
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--mode', default='info',type=str,help=None)
    parser.add_argument('--ROOT_DIR', default=None, type=str, help=DIR_HELP)
    parser.add_argument('--ROOT_DIR_MODE', default=None,help=None)
    parser.add_argument('--PROJECT_ID', default='pneu256_0', type=str, help=None)
    parser.add_argument('--model', default='resnet34', type=str, help=None)


    # PROJECT SETTINGS
    parser.add_argument('--img_size', default=256, type=int, help=None) 

    # training
    parser.add_argument('--n_iter', default=64, type=int, help=None) # we don't use epoch anymore
    parser.add_argument('--batch_size', default=8, type=int, help=None)
    parser.add_argument('--VALIDATION_ACC', default=0.8, type=float, help=None)
    parser.add_argument('--VALIDATION_FRACTION', default=0.2, type=float, help=None)
    parser.add_argument('--min_iter', default=12, type=int, help=None)

    # gax
    parser.add_argument('--method', default='Saliency', type=str, help=available_methods) 
    parser.add_argument('--split', default='test', type=str, help='train,test') 
    parser.add_argument('--label', default='NORMAL', type=str, help='NORMAL,PNEUMONIA') 
    parser.add_argument('--img_name', default=None, type=str, help=None) 
    parser.add_argument('--first_n_correct', default=4,type=int,help=None) 
    parser.add_argument('--submethod', default='sum', type=str, help=available_methods) 
    parser.add_argument('--gax_learning_rate', default=0.1, type=float, help=None) 
    parser.add_argument('--similarity_loss_factor', default=100, type=float, help=None)
    parser.add_argument('--target_co', default=10, type=float, help=None)

    # debug
    parser.add_argument('--n_debug', default=0, type=int, help=0) # how many data to load for debugging
    parser.add_argument('--show_batch', default=0, type=str, help=strbool_description) # string bool

    # utils
    parser.add_argument('--realtime_print', default=0, type=str, help=strbool_description)
    
    args = vars(parser.parse_args())  # is a dictionary
    # print(args)

    args['realtime_print'] = parse_bool_from_string(args['realtime_print'])
    args['show_batch'] = parse_bool_from_string(args['show_batch'])

    if args['ROOT_DIR'] is None:
        args['ROOT_DIR'] = os.getcwd()
    elif args['ROOT_DIR_MODE'] =='rel':
        os.chdir(os.path.join(os.getcwd(),args['ROOT_DIR']))
    elif args['ROOT_DIR_MODE'] == 'abs':
        os.chdir(args['ROOT_DIR'])


    if args['mode'] == 'info':
        print('PRINTING INFO, NO INFO YET.')
        exit()
    elif args['mode'] == 'data_reshuffle':
        from src.pneu.data import reshuffle
        reshuffle(args)
    elif args['mode'] == 'train':
        from src.pneu.train import Trainer
        tr = Trainer(args)
        tr.train()
    elif args['mode'] == 'evaluate':
        from src.pneu.train import Trainer
        tr = Trainer(args)
        tr.evaluate()
    elif args['mode'] == 'xai_display':
        from src.pneu.explain import GAX
        gax = GAX(args)
        gax.run_XAI_on_img()        
    elif args['mode'] == 'xai_collect':
        from src.pneu.explain import GAX
        gax = GAX(args)
        gax.collect_co_score_for_existing_methods()
    elif args['mode'] == 'xai_display_collection':
        from src.pneu.explain import GAX
        gax = GAX(args)
        gax.display_co_score_for_existing_methods()        
    elif args['mode'] == 'xai_display_boxplot':
        from src.pneu.explain import GAX
        gax = GAX(args)
        gax.display_co_score_boxplot_for_existing_methods()
    elif args['mode'] == 'gax':
        from src.pneu.explain import GAX
        gax = GAX(args)
        gax.gax()
    elif args['mode'] == 'gax_display':
        from src.pneu.explain import GAX
        gax = GAX(args)
        gax.display_gax_optimized_images()
