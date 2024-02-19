import os, joblib
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize

from .dnn import device
import torchvision.models as mod

def get_admission_th(L):
    return 0.5

def prep_data_and_dirs(dargs, modelname='model.pth'):
    print('prep_data_and_dirs...')

    from .imagenet_dict2 import ID2LABELS
    # folder_to_class_mapping = {f'{i}':i for i in range(10)}
    folder_to_class_mapping = { wnid:x[0] for wnid,x in ID2LABELS.items()}

    from src.settings.data_settings import IMAGENET_SETTING
    # rearrange_imagenet_data(IMAGENET_SETTING)

    ckptdir='checkpoint'
    dataname = 'imagenet'
    CKPT_DIR = os.path.join(ckptdir, dataname) 
    MODEL_DIR = os.path.join(CKPT_DIR, modelname)
    TRAIN_RESULT_DIR = MODEL_DIR+'.train.result'
    TEST_RESULT_DIR = MODEL_DIR+'.test.result'

    HYPER_FOLDER_DIR = os.path.join(CKPT_DIR, 'hyper')
    os.makedirs(CKPT_DIR,exist_ok=True)

    DIRS = {
        'TRAINING_DATA_DIR': os.path.join(IMAGENET_SETTING['SOURCE_DATA_DIR'], 'train'),
        # 'DATA_DIR' : IMAGENET_SETTING['DATA_DIR'],
        # 'TEST_DATA_DIR' : IMAGENET_SETTING['TEST_DATA_DIR'],
    
        'CKPT_DIR': CKPT_DIR,
        'MODEL_DIR': MODEL_DIR,        
        # 'TRAIN_RESULT_DIR': TRAIN_RESULT_DIR,
        # 'TEST_RESULT_DIR':TEST_RESULT_DIR,

        'HYPER_FOLDER_DIR':HYPER_FOLDER_DIR,

    }
    return folder_to_class_mapping, DIRS


def imagenet_reshape(img, input_size):
    s = img.shape
    if len(s)== 2:
        img = resize(img, input_size)
        img = np.array([img, img, img]) # set channel to 3
    elif len(s) == 3:
        img = resize(img, input_size + (3,))
        img = img.transpose(2,0,1)    
    return img

def prep_deep_neural_network_and_data_loader(dargs, parser, BOOLS, DIRS, device=None):
    dnn = mod.resnet18(weights=mod.ResNet18_Weights.DEFAULT, progress=False)
    dnn = dnn.to(device=device)
    dnn.eval()

    from .dnn import normalizeTransform

    def imagenet_img_loader(data_dir,input_size=(256,256)):
        pil_img = Image.open(data_dir)
        img = np.asarray(pil_img)/255.
        # print(img.shape, np.max(img), np.min(img))

        img = imagenet_reshape(img, input_size)

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device=device).to(torch.float)

        x = dnn( normalizeTransform(img) )
        x = x.clone().detach().cpu().numpy()
        x = x[0] # batch size=1, so we take the first item in the batch

        return x    
    return dnn, imagenet_img_loader