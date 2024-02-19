from xwsol.xresearch import compute_resnet50_scoremaps
from util import parse_bool_from_string, strbool_description
import argparse

if __name__=="__main__":


    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)
    parser.add_argument('--ROOT_DIR', default=None, type=str, help=None)
    # parser.add_argument('--scoremap_root', type=str,default='xunformattedlog',)
    parser.add_argument('--metadata_root', type=str, default='metadata/',help="Root folder of metadata.")
    parser.add_argument('--mask_root', type=str, default='dataset/',help="Root folder of masks (OpenImages).")
    parser.add_argument('--dataset_name', default="ILSVRC", type=str,help="One of [CUB, ILSVRC, OpenImages].")
    parser.add_argument('--split', type=str, default='test', help="One of [val, test]. They correspond to "
                             "train-fullsup and test, respectively.")
    # parser.add_argument('--scoremap_mode', type=str, default='random',help=None)
    # parser.add_argument('--scoremap_submode', type=str, default='unformatted',help=None) 
    parser.add_argument('--debug_toggles', default='000000',type=str,help='Only string of 0 and 1')
    parser.add_argument('--saved_imgs_idx', nargs='+', type=int, default=[0,1,2,3,4]) # for list args
    parser.add_argument('--DEBUG_N_ITER', default=100, type=str, help=None)
    args = parser.parse_args()
    args_dict = vars(args)  # is a dictionary



    args_dict['saved_imgs_idx'] = list(args_dict['saved_imgs_idx'])
    args_dict['scoremap_submode'] = 'unformatted'
    args_dict['for_eval'] = False

    scoremaps = ['saliency','gradcam','gbp', 'deeplift']
    # scoremaps = ['deeplift'] 
    for scoremap in scoremaps:
        args_dict['scoremap_mode'] = scoremap
        args_dict['scoremap_root'] = 'xunformattedlog/%s'%(str(scoremap))
        compute_resnet50_scoremaps(**args_dict, 
            DEBUG_TOGGLES=[parse_bool_from_string(x) for x in args_dict['debug_toggles']])


"""
python3 xquickruns_unformatted.py  --debug_toggles 100000  \
    --saved_imgs_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
        20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 \
        40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 \
        60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 \
        80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 



    print(list(range(100)))
    for i in range(100):
        if (i+1)%20==0:
            print(i)
        else:
            print(i,end= ' ')
    exit()
"""