"""
Extended the results from main forks with our own research.
"""
import argparse, os
from util import parse_bool_from_string, strbool_description
from data_loaders import configure_metadata
from data_loaders import get_image_ids

from evaluation import _get_cam_loader

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
import joblib
from os.path import exists as ospe
from os.path import join as ospj

import torch
import torch.nn as nn
from torchvision import transforms
from captum.attr import LayerGradCam, Saliency, DeepLift, LRP, GuidedBackprop

from xwsol.imagenet_dict import LABEL_DIR
from xwsol.xreformat import heatmap_reformat, tonumpy
from nbdt.utils import generate_checkpoint_fname
from util import get_bbr_cmap
""" 
Note: <heatmap_root> folder contains the heatmap files with the file names dictated by 
  the metadata/<dataset>/<split>/image_ids.txt files. If an image_id has slashes (/), e.g. val2/995/0.jpeg, 
  then the corresponding heatmaps shall be located at the corresponding sub-directories, 
  e.g. <heatmap_root>/val2/995/0.npy.

"""


def get_data_path(dataset_name, split):
    if dataset_name=='CUB':
        data_path = ospj('dataset', dataset_name, )
    elif dataset_name=='OpenImages':
        data_path = ospj('dataset', dataset_name, split)     
    elif dataset_name=='ILSVRC':
        data_path = ospj('dataset', dataset_name)     
    else:
        raise NotImplementedError()        

    return data_path


def compute_resnet50_scoremaps(DEBUG_TOGGLES='00000',**kwargs):
    """
    Set --NBDT 1 if we want to test the effect of model
      after training through the SoftTreeSupLoss for Neural Backed Decision Tree
    """


    if kwargs['DISABLE_GPU']:
        # YES. SOMETIMES WE NEED THIS FOR SILLY DEBUGGING........
        print('GPU is disabled for debugging...')
        device = 'cpu'
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print('trying to use CUDA...')    

    NBDT = parse_bool_from_string(int(kwargs['NBDT']))

    if not ospe(kwargs['scoremap_root']): os.makedirs(kwargs['scoremap_root'])
    meta_path = ospj(kwargs['metadata_root'], kwargs['dataset_name'], kwargs['split'])
    data_path = get_data_path(kwargs['dataset_name'], kwargs['split'])
    metadata = configure_metadata(meta_path)
    image_ids = get_image_ids(metadata)
    n_img = len(image_ids)
    print('meta_path:',meta_path) # e.g.  metadata/CUB/test
    DEBUG_N_ITER = kwargs['DEBUG_N_ITER']


    if kwargs['saved_imgs'] == 'auto':
        if DEBUG_TOGGLES[0]:
            kwargs['saved_imgs_idx'] = list(range(10))
        else:
            kwargs['saved_imgs_idx'] = list(range(100))


    for_eval = True if 'for_eval' not in kwargs else kwargs['for_eval']
    if not NBDT:
        hc = HeatmapComputerResnet50ILSVRC(**kwargs, device=device)
    else:
        hc = HeatmapComputerNbdtResnet50ILSVRC(**kwargs, device=device)

    import time
    start = time.time()
    for i,image_id in enumerate(image_ids):
        img_dir = ospj(data_path, image_id)
        pil_img = Image.open(img_dir).convert('RGB')
        img = np.asarray(pil_img)/255.
        h,w,c = img.shape

        for_display = True if i in kwargs['saved_imgs_idx'] else False

        if kwargs['scoremap_submode'] in ['input','unformatted'] :
            heatmap_dir = ospj(kwargs['scoremap_root'],image_id)
        elif kwargs['scoremap_submode'] in ['layer1','layer2','layer3','layer4']:
            heatmap_dir = ospj(kwargs['scoremap_root'] + '_' + kwargs['scoremap_submode'] ,image_id)
        else:
            raise NotImplementedError()            

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


        # plt.show()
        # plt.imshow(img)
        # plt.show()
        # exit()
        if i<10 or (i+1)==n_img:
            print(i,image_id)

        if (i+1)%16==0 or (i+1)==n_img:
            update_text = '%s/%s'%(str(i+1),str(n_img))
            print('%-64s'%(update_text), end='\r')
        if DEBUG_TOGGLES[0]:
            if (i+1)>=DEBUG_N_ITER:
                print('\nDebug early stopping at %s'%(str(i+1)))
                break

        torch.cuda.empty_cache()

    end = time.time()
    elapsed = end - start
    print('\n\ntime taken %s[s] '%(str(round(elapsed,1)), ))    

    print('\nDone generating heatmaps!')

class HeatmapComputerResnet50ILSVRC(object):
    def __init__(self, device=None, **kwargs):
        super(HeatmapComputerResnet50ILSVRC, self).__init__()
        print('HeatmapComputerResnet50ILSVRC ILSVRC type 1...') 

        self.device = device
        self.get_model()

        if kwargs['scoremap_mode'] in ['random', 'cam','gradcam', ]:
            pass
        elif kwargs['scoremap_mode'] in ['saliency','gbp', 'deeplift']:
            if kwargs['scoremap_submode'] in ['input','unformatted']:
                pass
            elif kwargs['scoremap_submode'] in ['layer1','layer2','layer3','layer4']:
                self.net.create_split_at_layer(split_layer_name=kwargs['scoremap_submode'])
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        normalizeTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        resize = transforms.Resize((224,224))
        self.normalizeImageTransform = transforms.Compose([ normalizeTransform, resize, ])
        try:
            xtest = self.normalizeImageTransform(torch.randn(size=(1,3,256,256)))
        except:
            # FOR OLDER VERSION
            self.normalizeImageTransform = transforms.Compose([transforms.ToPILImage(), resize, transforms.ToTensor(),normalizeTransform, ])
            print('ADJUSTING FOR OLDER PYTORCH VERSION!')

    def get_model(self):                    
        from xwsol.model import ResNet50CAM
        self.net = ResNet50CAM().to(device=self.device)
        self.net.eval()

    def predict_class(self, img, device=None):
        with torch.no_grad():
            x = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device=device).to(torch.float)
            try:
                x = self.normalizeImageTransform(x)
                y = self.net(x)
            except:
                # FOR OLDER VERSION
                x = self.normalizeImageTransform(x[0].cpu()).unsqueeze(0).to(device=device)
                y = self.net(x)
            y_pred = torch.argmax(y,axis=1) 
        return x,y,y_pred

    def forward_front(self, img, device=None):
        x = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device=device).to(torch.float)
        try:
            y1 = self.net.mod_front(self.normalizeImageTransform(x))
        except:
            # for older pytorch package
            y1 = self.net.mod_front(self.normalizeImageTransform(x[0].cpu()).unsqueeze(0).to(device=device))

        return y1

    # example how to generate and save heatmaps
    def save_random_heatmap(self, img, heatmap_dir, for_display):
        h,w,c = img.shape

        x,y,y_pred = self.predict_class(img)

        hmap = np.random.uniform(0.,1.,size=(h,w))
        np.save(heatmap_dir, hmap)
        if for_display: savefig(hmap, heatmap_dir, img, y_pred)

    #########################################################
    # Comparing standard heatmaps
    #########################################################

    def save_resnet50_cam(self, img, heatmap_dir, for_display=False, for_eval=True):
        # img is numpy array, shape (h,w,c)
        # it is supposed to be normalized to [0,1]
        h,w,c = img.shape

        x,y,y_pred = self.predict_class(img, device=self.device)

        hmap = self.net.compute_cam(x, labels=y_pred)
        hmap = heatmap_reformat(hmap.unsqueeze(0), h, w)
        # self.show_img(img,hmap, y_pred.item())

        if for_eval: np.save(heatmap_dir, hmap)
        if for_display: savefig(hmap, heatmap_dir, img, y_pred)


    def save_resnet50_saliency(self, img, heatmap_dir, for_display=False, for_eval=True, layer_target='input'):
        h,w,c = img.shape
        x,y,y_pred = self.predict_class(img, device=self.device)

        if layer_target=='input':        
            x.requires_grad = True
            try:
                # might run out of memory! (CUDA OUT OF MEMORY)
                saliency = Saliency(self.net, ) 
                hmap = saliency.attribute(x, y_pred, abs=False) # torch.Size([1, 3, 375, 500])
            except:
                self.net = self.net.to(device='cpu')
                saliency = Saliency(self.net, ) 
                hmap = saliency.attribute(x.to(device='cpu'), y_pred.to(device='cpu'), abs=False) # torch.Size([1, 3, 375, 500])
                self.net = self.net.to(device=self.device)

            hmap = heatmap_reformat(hmap, h, w, mode='saliency')
        elif layer_target in ['layer1', 'layer2', 'layer3', 'layer4']:
            y1 = self.forward_front(img, device=self.device).clone().detach()
            y1.requires_grad = True    
            try:
                saliency = Saliency(self.net.mod_back) 
                hmap = saliency.attribute(y1, y_pred, abs=False) # e.g. torch.Size([1, 2048, 12, 16])
            except:
                # might run out of memory! (CUDA OUT OF MEMORY)
                self.net = self.net.to(device='cpu')
                saliency = Saliency(self.net.mod_back) 
                hmap = saliency.attribute(y1.to(device='cpu'), y_pred.to(device='cpu'), abs=False) # e.g. torch.Size([1, 2048, 12, 16])
                self.net = self.net.to(device=self.device)

            hmap = heatmap_reformat(hmap, h, w, mode='saliency_alt')
        elif layer_target == 'unformatted':
            x.requires_grad = True
            saliency = Saliency(self.net, ) 
            hmap = saliency.attribute(x, y_pred, abs=False) # torch.Size([1, 3, 375, 500])
            hmap = heatmap_reformat(hmap, h, w, mode='unformatted')            
        else:
            raise NotImplementedError()

        if for_eval: np.save(heatmap_dir, hmap)
        if for_display: savefig(hmap, heatmap_dir, img, y_pred)


    def save_resnet50_gradcam(self, img, heatmap_dir, for_display=False, for_eval=True, layer_target='input') :
        # it is supposed to be normalized to [0,1]
        h,w,c = img.shape
        x,y,y_pred = self.predict_class(img, device=self.device)
        
        if layer_target=='input': 
            try:
                # might run out of memory! (CUDA OUT OF MEMORY)
                lgcam = LayerGradCam(self.net, self.net.backbone.conv1)
                hmap = lgcam.attribute(x, y_pred) # torch.Size([1, 1, 12, 16])
            except:
                self.net = self.net.to(device='cpu')
                lgcam = LayerGradCam(self.net, self.net.backbone.conv1)
                hmap = lgcam.attribute(x.to(device='cpu'), y_pred.to(device='cpu')) # torch.Size([1, 1, 12, 16])
                self.net = self.net.to(device=self.device)
                      
            hmap = heatmap_reformat(hmap, h, w,  mode='gradcam')
        elif layer_target in ['layer1', 'layer2', 'layer3', 'layer4']:
            try:
                lgcam = LayerGradCam(self.net, getattr(self.net.backbone,layer_target))
                hmap = lgcam.attribute(x, y_pred) # torch.Size([1, 1, 12, 16])
            except:
                self.net = self.net.to(device='cpu')
                lgcam = LayerGradCam(self.net, getattr(self.net.backbone,layer_target))
                hmap = lgcam.attribute(x.to(device='cpu'), y_pred.to(device='cpu')) # torch.Size([1, 1, 12, 16])  
                self.net = self.net.to(device=self.device)              
            hmap = heatmap_reformat(hmap, h, w,  mode='gradcam')
        elif layer_target == 'unformatted':
            lgcam = LayerGradCam(self.net, self.net.backbone.conv1)
            hmap = lgcam.attribute(x, y_pred) # torch.Size([1, 1, 12, 16])
            hmap = heatmap_reformat(hmap, h, w,  mode='unformatted')
        else:
            raise NotImplementedError()

        if for_eval: np.save(heatmap_dir, hmap)
        if for_display: savefig(hmap, heatmap_dir, img, y_pred)


    def save_resnet50_gbp(self, img, heatmap_dir, for_display=False, for_eval=True,  layer_target='input'):
        h,w,c = img.shape
        x,y,y_pred = self.predict_class(img, device=self.device)
        
        if layer_target=='input': 
            x.requires_grad = True
            gbp = GuidedBackprop(self.net)
            try:
                # might run out of memory! (CUDA OUT OF MEMORY)
                hmap = gbp.attribute(x, y_pred)
                hmap = heatmap_reformat(hmap, h, w,  mode='gbp')      
            except:
                self.net = self.net.to(device='cpu')
                hmap = gbp.attribute(x.to(device='cpu'), y_pred.to(device='cpu'))
                hmap = heatmap_reformat(hmap, h, w,  mode='gbp') 
                self.net = self.net.to(device=self.device)

        elif layer_target in ['layer1', 'layer2', 'layer3', 'layer4']:
            y1 = self.forward_front(img, device=self.device).clone().detach()
            y1.requires_grad = True     
            try:
                # might run out of memory! (CUDA OUT OF MEMORY)
                gbp = GuidedBackprop(self.net.mod_back, ) 
                hmap = gbp.attribute(y1, y_pred, ) 
            except:
                self.net = self.net.to(device='cpu')
                gbp = GuidedBackprop(self.net.mod_back, ) 
                hmap = gbp.attribute(y1.to(device='cpu'), y_pred.to(device='cpu'), ) 
                self.net = self.net.to(device=self.device)     
            hmap = heatmap_reformat(hmap, h, w, mode='gbp',layer=layer_target)
        elif layer_target == 'unformatted':
            x.requires_grad = True
            gbp = GuidedBackprop(self.net)
            hmap = gbp.attribute(x, y_pred)
            hmap = heatmap_reformat(hmap, h, w,  mode='unformatted')             
        else:
            raise NotImplementedError()

        if for_eval: np.save(heatmap_dir, hmap)
        if for_display: savefig(hmap, heatmap_dir, img, y_pred)


    def save_resnet50_deeplift(self, img, heatmap_dir,  for_display=False, for_eval=True,  layer_target='input'):
        h,w,c = img.shape
        x,y,y_pred = self.predict_class(img, device=self.device)
        
        if layer_target=='input': 
            x.requires_grad=True
            try:
                # might run out of memory! (CUDA OUT OF MEMORY)
                dlift = DeepLift(self.net)
                hmap = dlift.attribute(x, target=y_pred)
            except:
                self.net = self.net.to(device='cpu')
                dlift = DeepLift(self.net)
                hmap = dlift.attribute(x.to(device='cpu'), target=y_pred.to(device='cpu'))   
                self.net = self.net.to(device=self.device)

            hmap = heatmap_reformat(hmap, h, w,  mode='deeplift') 
        elif layer_target in ['layer1', 'layer2', 'layer3', 'layer4']:
            y1 = self.forward_front(img, device=self.device).clone().detach()    
            y1.requires_grad = True
            try:
                # might run out of memory! (CUDA OUT OF MEMORY)    
                dlift = DeepLift(self.net.mod_back, ) 
                hmap = dlift.attribute(y1, target=y_pred) 
            except:
                self.net = self.net.to(device='cpu')
                dlift = DeepLift(self.net.mod_back, ) 
                hmap = dlift.attribute(y1.to(device='cpu'), target=y_pred.to(device='cpu')) 
                self.net = self.net.to(device=self.device)

            hmap = heatmap_reformat(hmap, h, w, mode='deeplift_alt')
        elif layer_target == 'unformatted':
            x.requires_grad=True
            dlift = DeepLift(self.net)
            hmap = dlift.attribute(x, target=y_pred) 
            hmap = heatmap_reformat(hmap, h, w,  mode='unformatted')             
        else:
            raise NotImplementedError()

        if for_eval: np.save(heatmap_dir, hmap)  
        if for_display: savefig(hmap, heatmap_dir, img, y_pred)


    # #################################################
    # Others
    #################################################

    def show_img(self,img,hmap, y_pred):
        # img is numpy array, shape (h,w,c)
        # hmap is (h,w)

        plt.figure()
        plt.gcf().add_subplot(111)
        plt.gca().imshow(img)
        plt.gca().imshow(hmap, alpha=0.5)
        plt.title('%s'%(str(y_pred)))
        plt.show()
        plt.close()
        exit()

class HeatmapComputerNbdtResnet50ILSVRC(HeatmapComputerResnet50ILSVRC):
    # Same as HeatmapComputerResnet50ILSVRC except NBDT version is used for prediction

    def __init__(self, device=None, **kwargs):
        self.kwargs = kwargs
        super(HeatmapComputerNbdtResnet50ILSVRC, self).__init__(**kwargs, device=device)

    def get_model(self):
        print('Using NBDT...!')                
        from xwsol.model import ResNet50CAM
        self.net = ResNet50CAM().to(device=self.device)
        
        checkpoint_fname = generate_checkpoint_fname(**self.kwargs)
        try:
            state = torch.load(f"./checkpoint/{checkpoint_fname}.pth",map_location=self.device)
        except:
            # if checkpoint is from older pytorch version
            state = torch.jit.load(f"./checkpoint/{checkpoint_fname}.pth")


        def load_state_dict(net, state_dict):
            ####################################
            # THIS FUNCTION IS FROM NBDT. Their arrangement is a little quirky so they need this.
            try:
                net.load_state_dict(state_dict)
            except RuntimeError as e:
                if "Missing key(s) in state_dict:" in str(e):
                    net.load_state_dict(
                        {
                            key.replace("module.", "", 1): value
                            for key, value in state_dict.items()
                        }
                    )
            return net
            ####################################

        """
        state = {
            "net": net.state_dict(),"acc": acc,"epoch": epoch,
        }
        """
        try:
            self.net.backbone.load_state_dict(state['net'])
        except:
            print('loading NBDT-trained model using their fix from nbdt.models.utils')
            self.net.backbone = load_state_dict(self.net.backbone, state['net'])
        self.net.eval()

    def predict_class(self, img, device=None):
        x = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device=device).to(torch.float)
        try:
            y = self.net(self.normalizeImageTransform(x))
        except:
            # FOR OLDER VERSION
            y = self.net(self.normalizeImageTransform(x[0].cpu()).unsqueeze(0).to(device=device))
        y_pred = torch.argmax(y,axis=1) 
        return x,y,y_pred



# run this function after computing scoremaps for evaluation
# def evaluate_wsol(scoremap_root, metadata_root, mask_root, dataset_name, split,
#                   multi_contour_eval, multi_iou_eval, iou_threshold_list,
#                   cam_curve_interval=.001):

def show_results(**kwargs):
    RESULT_DIR = ospj(kwargs['scoremap_root'], 'score.result')
    print(RESULT_DIR)

    results = joblib.load(RESULT_DIR)
    print(results)
    
    """
    results:
      each self.num_correct[_THRESHOLD]: 
        len == len(self.cam_threshold_list)
    """

def offticks():
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

def savefig(hmap, heatmap_dir, img, y_pred, ):
    label = str(y_pred.item()) # y_pred is a tensor
    this_label = LABEL_DIR[label]["label"]
    if len(this_label)>18:
        this_label = this_label[:18]+'...'
    plt.figure(figsize=(8,4))

    
    plt.gcf().add_subplot(221)
    plt.gca().imshow(img)
    im = plt.gca().imshow(hmap, vmin=0, vmax=1, alpha=0.5)
    plt.colorbar(im,fraction=0.03, pad=0.04)
    # offticks()
    plt.title('%s : %s'%(str(label),str(this_label)))
    

    plt.gcf().add_subplot(222)
    im2 = plt.gca().imshow(hmap, vmin=0, vmax=1)
    offticks()
    plt.colorbar(im2,fraction=0.03, pad=0.04)


    if np.abs(np.mean(hmap))>0.1:
        hmap = hmap - np.mean(hmap)

    plt.gcf().add_subplot(223)
    plt.gca().imshow(img)
    im = plt.gca().imshow(hmap, vmin=-1, vmax=1, alpha=0.5, cmap=get_bbr_cmap())
    plt.colorbar(im,fraction=0.03, pad=0.04)

    plt.gcf().add_subplot(224)
    im2 = plt.gca().imshow(hmap, vmin=-1, vmax=1, cmap=get_bbr_cmap())
    offticks()
    plt.colorbar(im2,fraction=0.03, pad=0.04)


    plt.tight_layout()
    plt.savefig(heatmap_dir)
    plt.close()

def mkdir_if_not_exist(heatmap_dir):
    hmap_folder = os.path.join(*heatmap_dir.split('/')[:-1])
    if not os.path.exists(hmap_folder):
        os.makedirs(hmap_folder)
