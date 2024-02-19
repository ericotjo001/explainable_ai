import argparse


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)
    parser.add_argument('--ROOT_DIR', default=None, type=str, help=None)
    args, unknown = parser.parse_known_args()
    args_dict = vars(args)  # is a dictionary

    print('Buffer just in case....\n\n\nExecuting quickruns...')

    if args_dict['ROOT_DIR'] is not None:
        print('switching root dir to ', args_dict['ROOT_DIR'])
        import os        
        os.chdir(args_dict['ROOT_DIR'])
        print(os.getcwd())



    from util import parse_bool_from_string, strbool_description
    from config import str2bool
    from evaluation import evaluate_wsol
    from xwsol.xresearch import compute_resnet50_scoremaps
    from xwsol.xresult import collate_results, compare_with_nbdt


    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)
    parser.add_argument('--ROOT_DIR', default=None, type=str, help=None)
    parser.add_argument('--mode', default='scoremap_and_eval', type=str, help=None)
    parser.add_argument('--scoremap_root', type=str,default='train_log/scoremaps',
        help="The root folder for score maps to be evaluated. Ignore any / at the end.")
    parser.add_argument('--metadata_root', type=str, default='metadata/',help="Root folder of metadata.")
    parser.add_argument('--mask_root', type=str, default='dataset/',help="Root folder of masks (OpenImages).")
    parser.add_argument('--dataset_name', default="ILSVRC", type=str,help="One of [CUB, ILSVRC, OpenImages].")
    parser.add_argument('--split', type=str, default='test', help="One of [val, test]. They correspond to "
                             "train-fullsup and test, respectively.")

    parser.add_argument('--scoremap_mode', type=str, default='random',help=None)
    parser.add_argument('--scoremap_submode', type=str, default='input',help=None)
    parser.add_argument('--cam_curve_interval', type=float, default=0.01, help="At which threshold intervals will the score maps be evaluated?.")
    parser.add_argument('--multi_contour_eval', type=str2bool, nargs='?',const=True, default=False)
    parser.add_argument('--multi_iou_eval', type=str2bool, nargs='?',const=True, default=False)
    parser.add_argument('--iou_threshold_list', nargs='+',type=int, default=[30, 50, 70])

    ################################
    # NBDT related arguments
    ################################
    parser.add_argument("--arch", default="ResNet50CAM", help='DO NOT CHANGE.')
    parser.add_argument("--loss", default=["SoftTreeSupLoss"], nargs="+")
    # parser.add_argument("--hierarchy",help="Hierarchy to use. If supplied, will be used to "
    #     "generate --path-graph. --path-graph takes precedence.",)
    parser.add_argument("--path-graph", help="Path to graph-*.json file.")  # WARNING: hard-coded suffix -build in generate_checkpoint_fname
    parser.add_argument("--path-wnids", help="Path to wnids.txt file.")
    parser.add_argument("--dataset", default="ILSVRC") # we only deal with Imagenet1000 for now

    parser.add_argument('--saved_imgs', type=str, default='auto') 
    parser.add_argument('--saved_imgs_idx', nargs='+',type=int, default=[0,1,2,3,4]) # for list args
    parser.add_argument('--debug_toggles', default='000000',type=str,help='Only string of 0 and 1')
    parser.add_argument('--DEBUG_N_ITER', default=16, type=int, help=None)


    ################################
    # Final Results Related
    ################################
    parser.add_argument('--scoremap_root_compare', type=str,default='train_log/scoremaps_nbdt',
        help="For final results, compare the log folder of WSOL results and its NBDT version.")


    BOOLS = { # see strbool_description
        #####################################################################################
        # IT IS VERY IRRITATING BUT YES DISABLE GPU, SOMETIMES YOU WILL NEED IT WHEN YOUR ARE DEBUGGING
        # YOUR LAPTOP AND INSTALLATIONS SOMETIMES AREN'T COMPATIBLE WITH THE SERVER STUFF
        #####################################################################################
        'DISABLE_GPU': "0",
        'NBDT': "0",
    }
    for bkey,b in BOOLS.items():
        parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)


    # parse again
    args = parser.parse_args()
    while args.scoremap_root[-1] == '/':
         args.scoremap_root = args.scoremap_root[:-1]
    args_dict = vars(args)  # is a dictionary
    args_dict['saved_imgs_idx'] = list(args_dict['saved_imgs_idx'])

    for b in BOOLS:
        args_dict[b] = parse_bool_from_string(args_dict[b])
        setattr(args, b, args_dict[b])

    if args_dict['mode'] == 'scoremap_and_eval':
        compute_resnet50_scoremaps(**args_dict, 
            DEBUG_TOGGLES=[parse_bool_from_string(x) for x in args_dict['debug_toggles']])

        def eval(scoremap_root):
            evaluate_wsol(scoremap_root=scoremap_root,
                          metadata_root=args.metadata_root,
                          mask_root=args.mask_root,
                          dataset_name=args.dataset_name,
                          split=args.split,
                          cam_curve_interval=args.cam_curve_interval,
                          multi_contour_eval=args.multi_contour_eval,
                          multi_iou_eval=args.multi_iou_eval,
                          iou_threshold_list=args.iou_threshold_list,
                          DEBUG_TOGGLES=args.debug_toggles)

        eval_folder = args.scoremap_root
        if not args.scoremap_submode == 'input':
            eval_folder = eval_folder + '_%s'%(str(args.scoremap_submode)) 
        eval(eval_folder)
    elif args_dict['mode'] == 'collate_results':
        collate_results(args_dict)
    elif args_dict['mode'] == 'compare_results':
        compare_with_nbdt(args_dict)
    elif args_dict['mode'] == 'displaybbox':
         from xwsol.xdisplay import displaybbox
         displaybbox(**args_dict)
    else:
        raise NotImplementedError('Invalid mode.')



"""
#########################################################
# standard heatmaps FOR COMPARISON
#########################################################

python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_cam \
    --scoremap_mode cam  --debug_toggles 100000  


#########################################################
# Without NBDT
#########################################################
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency \
    --scoremap_mode saliency --scoremap_submode input  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency \
    --scoremap_mode saliency --scoremap_submode layer1  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency \
    --scoremap_mode saliency --scoremap_submode layer2  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency \
    --scoremap_mode saliency --scoremap_submode layer3  --debug_toggles 100000  

python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gradcam \
    --scoremap_mode gradcam --scoremap_submode input  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gradcam \
    --scoremap_mode gradcam --scoremap_submode layer1  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gradcam \
    --scoremap_mode gradcam --scoremap_submode layer2  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gradcam \
    --scoremap_mode gradcam --scoremap_submode layer3  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gradcam \
    --scoremap_mode gradcam --scoremap_submode layer4  --debug_toggles 100000 

python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gbp \
    --scoremap_mode gbp --scoremap_submode input  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gbp \
    --scoremap_mode gbp --scoremap_submode layer1  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gbp \
    --scoremap_mode gbp --scoremap_submode layer2  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gbp \
    --scoremap_mode gbp --scoremap_submode layer3  --debug_toggles 100000  

python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_deeplift \
    --scoremap_mode deeplift --scoremap_submode input  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_deeplift \
    --scoremap_mode deeplift --scoremap_submode layer1  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_deeplift \
    --scoremap_mode deeplift --scoremap_submode layer2  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_deeplift \
    --scoremap_mode deeplift --scoremap_submode layer3  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_deeplift \
    --scoremap_mode deeplift --scoremap_submode layer4  --debug_toggles 100000  


EXCLUDE MODE COLLAPSE python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency \
    --scoremap_mode saliency --scoremap_submode layer4  --debug_toggles 100000  
EXCLUDE MODE COLLAPSE python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gbp \
    --scoremap_mode gbp --scoremap_submode layer4  --debug_toggles 100000  

#########################################################
# NBDT
#########################################################

python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency_NBDT --NBDT 1 \
    --scoremap_mode saliency --scoremap_submode input  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency_NBDT --NBDT 1\
    --scoremap_mode saliency --scoremap_submode layer1  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency_NBDT --NBDT 1\
    --scoremap_mode saliency --scoremap_submode layer2  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency_NBDT --NBDT 1\
    --scoremap_mode saliency --scoremap_submode layer3  --debug_toggles 100000  

python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gradcam_NBDT --NBDT 1\
    --scoremap_mode gradcam --scoremap_submode input  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gradcam_NBDT --NBDT 1\
    --scoremap_mode gradcam --scoremap_submode layer1  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gradcam_NBDT --NBDT 1\
    --scoremap_mode gradcam --scoremap_submode layer2  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gradcam_NBDT --NBDT 1\
    --scoremap_mode gradcam --scoremap_submode layer3  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gradcam_NBDT --NBDT 1\
    --scoremap_mode gradcam --scoremap_submode layer4  --debug_toggles 100000  

python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gbp_NBDT --NBDT 1\
    --scoremap_mode gbp --scoremap_submode input  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gbp_NBDT --NBDT 1\
    --scoremap_mode gbp --scoremap_submode layer1  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gbp_NBDT --NBDT 1\
    --scoremap_mode gbp --scoremap_submode layer2  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_gbp_NBDT --NBDT 1\
    --scoremap_mode gbp --scoremap_submode layer3  --debug_toggles 100000  

python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_deeplift_NBDT --NBDT 1\
    --scoremap_mode deeplift --scoremap_submode input  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_deeplift_NBDT --NBDT 1\
    --scoremap_mode deeplift --scoremap_submode layer1  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_deeplift_NBDT --NBDT 1\
    --scoremap_mode deeplift --scoremap_submode layer2  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_deeplift_NBDT --NBDT 1\
    --scoremap_mode deeplift --scoremap_submode layer3  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_deeplift_NBDT --NBDT 1\
    --scoremap_mode deeplift --scoremap_submode layer4  --debug_toggles 100000  

#########################################################
Test on Singularity
#########################################################
singularity exec --nv --bind wsolevaluation-master/:/mnt MyPyTorchSandBox/ python3 /mnt/xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_cam \
    --scoremap_mode cam  --DISABLE_GPU 1 --ROOT_DIR wsolevaluation-master --debug_toggles 100000 
singularity exec --nv --bind wsolevaluation-master/:/mnt MyPyTorchSandBox/ python3 /mnt/xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency \
    --scoremap_mode saliency --scoremap_submode input  --DISABLE_GPU 1 --ROOT_DIR wsolevaluation-master --debug_toggles 100000 
singularity exec --nv --bind wsolevaluation-master/:/mnt MyPyTorchSandBox/ python3 /mnt/xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency \
    --scoremap_mode saliency --scoremap_submode layer1  --DISABLE_GPU 1 --ROOT_DIR wsolevaluation-master --debug_toggles 100000 
singularity exec --nv --bind wsolevaluation-master/:/mnt MyPyTorchSandBox/ python3 /mnt/xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency_NBDT --NBDT 1 \
    --scoremap_mode saliency --scoremap_submode input  --DISABLE_GPU 1 --ROOT_DIR wsolevaluation-master --debug_toggles 100000 

#########################################################
 RESULTS 
#########################################################

python3 xquickruns_resnet50.py --mode collate_results --scoremap_root xresearchlog.resnet50.nscc.1
python3 xquickruns_resnet50.py --mode collate_results --scoremap_root xresearchlog.resnet50.nscc.2 --NBDT 1
python xquickruns_resnet50.py --mode compare_results --scoremap_root xresearchlog.resnet50.nscc.1 --scoremap_root_compare xresearchlog.resnet50.nscc.

"""