import argparse
from util import strbool_description , parse_bool_from_string

"""
Step 1. Induce the graph for training
python3 x_nbdt_main.py --mode inducegraph --arch ResNet18 

python3 x_nbdt_main.py --mode inducegraph --arch ResNet50CAM 

Step 2. Train with TreeSupLoss
to observe the data (no training), use this (first debug toggle is on)
python3 x_nbdt_main.py --mode train --arch ResNet50 --hierarchy induced-ResNet50  --epochs 1 --batch-size 4 --debug_toggles 1000000

Training starts!
python3 x_nbdt_main.py --mode train --arch ResNet50CAM --hierarchy induced-ResNet50CAM --epochs 2 --print_every 1 --batch-size 4 --resume 0 --eval 0 --lr 0.0001 --loss SoftTreeSupLoss --debug_toggles 0100000

python3 x_nbdt_main.py --mode train --arch ResNet50CAM --hierarchy induced-ResNet50CAM --epochs 2 --print_every 1 --batch-size 4 --resume 1 --eval 1 --lr 0.0001 --loss SoftTreeSupLoss --debug_toggles 0100000


Test on singularity
singularity exec --nv --bind wsolevaluation-master/:/mnt MyPyTorchSandBox/ python3 /mnt/x_nbdt_main.py --mode train --arch ResNet50CAM --hierarchy induced-ResNet50CAM     --epochs 1 --batch-size 4 --ROOT_DIR wsolevaluation-master --debug_toggles 0100000 --DISABLE_GPU 1
"""


if __name__=='__main__':
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

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)
    parser.add_argument('--mode', default=None, type=str, help=None)
    parser.add_argument('--submode', default=None, type=str, help=None)

    parser.add_argument("--dataset", default="ILSVRC") # we only deal with Imagenet1000 for now
    parser.add_argument("--arch", default="ResNet18", )
    parser.add_argument("--path-resume", default="", help="Overrides checkpoint path generation")
    parser.add_argument('--ROOT_DIR', default=None, type=str, help=None)
    # parser.add_argument('--PROJECT_NAME', default='srd_project', type=str, help=None)

    parser.add_argument("--hierarchy",help="Hierarchy to use. If supplied, will be used to "
        "generate --path-graph. --path-graph takes precedence.",)
    parser.add_argument("--path-graph", help="Path to graph-*.json file.")  # WARNING: hard-coded suffix -build in generate_checkpoint_fname
    parser.add_argument("--path-wnids", help="Path to wnids.txt file.")

    parser.add_argument("--epochs","-e",default=200,type=int,help="By default, lr schedule is scaled accordingly",)
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size used for training")
    parser.add_argument("--loss", default=["SoftTreeSupLoss"], nargs="+")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--metric",  default="top1")
    parser.add_argument("--analysis",  help="Run analysis after each epoch")
    parser.add_argument("--disable-test-eval",help="Allows you to run model inference on a test dataset "
        " different from train dataset. Use an anlayzer to define "
        "a metric.",
        action="store_true",
    )

    
    parser.add_argument('--print_every', default=256, type=int, help=None)

    from nbdt import loss
    loss.add_arguments(parser)

    BOOLS = { # see strbool_description
        'resume': 0,
        'eval':0,
        
        #####################################################################################
        # IT IS VERY IRRITATING BUT YES DISABLE GPU, SOMETIMES YOU WILL NEED IT WHEN YOUR ARE DEBUGGING
        # YOUR LAPTOP AND INSTALLATIONS SOMETIMES AREN'T COMPATIBLE WITH THE SERVER STUFF
        #####################################################################################
        'DISABLE_GPU':0,
    }
    for bkey,b in BOOLS.items():
        parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)
    parser.add_argument('--debug_toggles', default='0000000',type=str,help='Only string of 0 and 1') 
    # TOGGLES = [parse_bool_from_string(x) for x in args['debug_toggles']]   

    args = parser.parse_args()
    # dargs = vars(args)  # is a dictionary

    for bkey,b in BOOLS.items():
        # print(bkey, getattr(args,bkey))
        setattr(args, bkey, parse_bool_from_string(getattr(args,bkey)))

    # See results here
    # print(args)
    # if parse_bool_from_string(args['BOOL']): print('true')
    # if not parse_bool_from_string(args['BOOL2']): print('also true')

    # TOGGLES = [parse_bool_from_string(x) for x in args['debug_toggles']]
    # print(TOGGLES

    if args.mode == 'inducegraph':
        from nbdt.xinduce import inducegraph
        inducegraph(args)
    elif args.mode == 'train':
        from nbdt.xtrain import training_entry_pipeline
        training_entry_pipeline(args)
    else:
        print('No valid mode selected.')



