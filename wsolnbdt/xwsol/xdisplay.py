
import argparse, os
from util import parse_bool_from_string, strbool_description
from data_loaders import configure_metadata
from data_loaders import get_image_ids

import torch

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
import joblib
from os.path import exists as ospe
from os.path import join as ospj

from xwsol.xresearch import mkdir_if_not_exist, get_data_path

DISPLAY_IDXS = range(100)


def displaybbox(**kwargs):
    print('displaybbox!')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NBDT = parse_bool_from_string(int(kwargs['NBDT']))

    if not NBDT:
        from xwsol.xresearch import HeatmapComputerResnet50ILSVRC
        hc = HeatmapComputerResnet50ILSVRC(**kwargs, device=device)
    else:
        print('using NBDT!')
        from xwsol.xresearch import HeatmapComputerNbdtResnet50ILSVRC
        hc = HeatmapComputerNbdtResnet50ILSVRC(**kwargs, device=device)

    if not ospe(kwargs['scoremap_root']): os.makedirs(kwargs['scoremap_root'])
    meta_path = ospj(kwargs['metadata_root'], kwargs['dataset_name'], kwargs['split'])
    data_path = get_data_path(kwargs['dataset_name'], kwargs['split'])
    metadata = configure_metadata(meta_path)
    image_ids = get_image_ids(metadata)

    dataset_name = kwargs['dataset_name']
    assert(dataset_name == 'ILSVRC') # other dataset not yet available


    for idx in DISPLAY_IDXS:
        current_img_id = image_ids[int(idx)]
        img_dir = ospj(data_path, current_img_id)
        pil_img = Image.open(img_dir).convert('RGB')
        img = np.asarray(pil_img)/255.
        h,w,c = img.shape

        for_display = False
        for_eval = True

        heatmap_dir = os.path.join('xdisplaylog',kwargs['scoremap_mode'] + '_' + kwargs['scoremap_submode'],current_img_id) 
        mkdir_if_not_exist(heatmap_dir)

        if kwargs['scoremap_mode'] =='random':
            hc.save_random_heatmap(img, heatmap_dir, for_display=for_display, for_eval=for_eval)
        elif kwargs['scoremap_mode'] == 'cam':
            hc.save_resnet50_cam(img, heatmap_dir, for_display=for_display, for_eval=for_eval)
        elif kwargs['scoremap_mode'] == 'saliency':
            hc.save_resnet50_saliency(img, heatmap_dir, for_display=for_display, for_eval=for_eval, 
                layer_target=kwargs['scoremap_submode']) 
        elif kwargs['scoremap_mode'] == 'gradcam':
            hc.save_resnet50_gradcam(img, heatmap_dir, for_display=for_display, for_eval=for_eval, 
                layer_target=kwargs['scoremap_submode'])      
        elif kwargs['scoremap_mode'] == 'gbp':
            hc.save_resnet50_gbp(img, heatmap_dir, for_display=for_display, for_eval=for_eval, 
                layer_target=kwargs['scoremap_submode'])
        elif kwargs['scoremap_mode'] == 'deeplift':
            hc.save_resnet50_deeplift(img, heatmap_dir, for_display=for_display, for_eval=for_eval, 
                layer_target=kwargs['scoremap_submode'])
        else:
            raise NotImplementedError()

        scoremap = np.load(heatmap_dir+'.npy').astype(np.float)
        cascading_bboxmap = heatmap_dir + '.cbbox.png'

        cam_threshold_list = list(np.arange(0, 1, kwargs['cam_curve_interval']))
        from evaluation import compute_bboxes_from_scoremaps
        estimated_boxes_at_each_thr, number_of_box_list = compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list=cam_threshold_list , multi_contour_eval=False)

        n = len(cam_threshold_list)
        n2 = int(n/2)
        boxes_at_thresholds = np.concatenate(estimated_boxes_at_each_thr, axis=0)
        # print(boxes_at_thresholds.shape) # (n, [x0, y0, x1, y1]) where n is len(cam_threshold_list)
        # print(boxes_at_thresholds)


        plt.figure()
        plt.gcf().add_subplot(121)
        plt.gca().imshow(img)
        for i in range(n):
            this_box = boxes_at_thresholds[i]
            x0,y0 = this_box[0], this_box[1]
            w,h = this_box[2] -x0, this_box[3] -y0
            plt.gca().add_patch(plt.Rectangle((x0,y0),w,h, facecolor=(i/n,(i+n2)/(2*n),(n-i)/n), alpha=0.01))
    
        plt.gcf().add_subplot(122)
        plt.gca().imshow(img, alpha=0.3)
        for i in range(n):
            this_box = boxes_at_thresholds[i]
            x0,y0 = this_box[0], this_box[1]
            w,h = this_box[2] -x0, this_box[3] -y0
            plt.gca().add_patch(plt.Rectangle((x0,y0),w,h,edgecolor=(i/n,(i+n2 )/(2*n),(n-i)/n), fill=False, alpha=0.5))
        plt.title(f'{kwargs["scoremap_mode"]} {kwargs["scoremap_submode"]}')

        plt.savefig(cascading_bboxmap)
        plt.close()
        os.remove(heatmap_dir+'.npy')

"""

python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode saliency --scoremap_submode input 
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode saliency --scoremap_submode layer1
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode saliency --scoremap_submode layer2
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode saliency --scoremap_submode layer3

python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode gradcam --scoremap_submode input 
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode gradcam --scoremap_submode layer1
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode gradcam --scoremap_submode layer2
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode gradcam --scoremap_submode layer3
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode gradcam --scoremap_submode layer4

python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode gbp --scoremap_submode input 
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode gbp --scoremap_submode layer1
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode gbp --scoremap_submode layer2
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode gbp --scoremap_submode layer3

python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode deeplift --scoremap_submode input 
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode deeplift --scoremap_submode layer1
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode deeplift --scoremap_submode layer2
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode deeplift --scoremap_submode layer3
python3 xquickruns_resnet50.py --mode displaybbox --scoremap_mode deeplift --scoremap_submode layer4

"""