import argparse, os
from utils import *

"""
my data dir: --DATA_DIR "D:/shigoto/ImageNet/ILSVRC"
"""

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
    parser.add_argument('--model', default='resnet34',type=str,help=None)
    parser.add_argument('--ROOT_DIR', default=None, type=str, help=DIR_HELP)
    parser.add_argument('--ROOT_DIR_MODE', default=None,help=None)
    parser.add_argument('--DATA_DIR', default=None, type=str, help=None)
    parser.add_argument('--PROJECT_ID', default='imgnet256_0', type=str, help=None)

    # PROJECT SETTINGS
    parser.add_argument('--img_size', default=256, type=int, help=None) 

    # TRAINING
    parser.add_argument('--n_iter', default=16,type=int,help=None)
    parser.add_argument('--batch_size', default=4,type=int,help=None)   

    # other settings 
    parser.add_argument('--method', default='Saliency', type=str, help=None) # 'Saliency,InputXGradient, LayerGradCam, Deconvolution, GuidedBackprop, DeepLift' 
    parser.add_argument('--split', default='val', type=str, help='train,val') 
    parser.add_argument('--selected_img_name', default=None, type=str, help=None) # for eval_selected_image

    # GAX
    parser.add_argument('--img_index', default=None,type=int,help=None) # choose which image to apply GAX on. If None, get the first_n_correct
    parser.add_argument('--first_n_correct', default=4,type=int,help=None) 
    parser.add_argument('--submethod', default='sum', type=str, help=available_methods) 
    parser.add_argument('--gax_learning_rate', default=1e-4, type=float, help=None) 
    parser.add_argument('--similarity_loss_factor', default=100, type=float, help=None)
    parser.add_argument('--target_co', default=10, type=float, help=None) 
    parser.add_argument('--PROJECT_IDs', nargs='+', default=['resnet34','alexnet'])


    # debug
    parser.add_argument('--n_debug_imagenet', default=108,type=int,help='set to 0 to get all data')


    args = vars(parser.parse_args())  # is a dictionary

    bool_args = []
    for this_bool in bool_args:
        args[this_bool] = parse_bool_from_string(args[this_bool])

    if args['ROOT_DIR'] is None:
        args['ROOT_DIR'] = os.getcwd()
    
    ROOT_DIR =  args['ROOT_DIR']
    ROOT_DIR_MODE = args['ROOT_DIR_MODE']
    if ROOT_DIR_MODE =='rel':
        os.chdir(os.path.join(os.getcwd(),ROOT_DIR))
    elif ROOT_DIR_MODE == 'abs':
        os.chdir(ROOT_DIR)

    if args['mode'] == 'info':
        print('PRINTING INFO, NO INFO YET.')
        exit()
    elif args['mode'] == 'train':
        from src.imgnet.train import Trainer
        tr = Trainer(args)
        tr.train()
    elif args['mode'] == 'xai_collect':
        from src.imgnet.explain import GAX
        gax = GAX(args)
        gax.collect_co_score_for_existing_methods()
    elif args['mode'] == 'xai_display_collection':
        from src.imgnet.explain import GAX
        gax = GAX(args)
        gax.display_co_score_for_existing_methods()  
    elif args['mode'] == 'xai_display_boxplot':
        from src.imgnet.explain import GAX
        gax = GAX(args)
        gax.display_co_score_boxplot_for_existing_methods()
    elif args['mode'] == 'gax':
        from src.imgnet.explain import GAX
        gax = GAX(args)
        gax.gax()

    elif args['mode'] == 'gax_display':
        from src.imgnet.explain import GAX
        gax = GAX(args)
        gax.display_gax_optimized_images()

    elif args['mode'] == 'gax_coplot':
        from src.imgnet.explain import GAX
        gax = GAX(args)
        gax.mass_plot_co_scores()

    elif args['mode'] == 'eval_selected_image':
        from src.imgnet.train import Trainer
        tr = Trainer(args)
        tr.eval_selected_image()

    elif args['mode'] == 'xai_collect2':
        from src.imgnet.explain2 import GAX2
        gax = GAX2(args)
        gax.collect_co_score_for_layer_comparison_and_display()
    elif args['mode'] == 'xai_display_boxplot2':
        from src.imgnet.explain2 import GAX2
        gax = GAX2(args)
        gax.display_co_score_boxplot_for_layer_comparison()