import os, shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .utils import manage_directories, resize_numpy_img_array

import torch
from torch.utils.data import Dataset, DataLoader


def reshuffle(args):
    print('reshuffle')
    DIRS = manage_directories(args)
    TRAIN_DATA_DIR = os.path.join(DIRS['DATA_DIR'],'train')
    VAL_DATA_DIR = os.path.join(DIRS['DATA_DIR'],'val')

    for label in ['NORMAL','PNEUMONIA']:
        train_dir = os.path.join(TRAIN_DATA_DIR, label)
        val_dir = os.path.join(VAL_DATA_DIR, label)
        print(train_dir, ' to ', val_dir)

        train_imgs = os.listdir(train_dir)
        for_validation = np.random.choice([0,1], len(train_imgs), 
            p=[1-args['VALIDATION_FRACTION'],args['VALIDATION_FRACTION']])

        val_indices = np.where(for_validation==1)
        # print(len(train_imgs), np.sum(for_validation))
        # print(val_indices)
        for i in val_indices[0]:
            img_src_dir = os.path.join(train_dir,train_imgs[i]) # cut
            img_tgt_dir = os.path.join(val_dir, train_imgs[i]) # paste
            shutil.move(img_src_dir, img_tgt_dir)

class ImageProcessor():
    def __init__(self, resize=(960,960)):
        super(ImageProcessor, self).__init__()
        self.target_size = (3,) + resize

    def simple_img_preprocessing(self, pil_img):
        img = np.asarray(pil_img) 
        # self.show_img(img)
        if len(img.shape)==2: 
            # print(img.shape) # only H, W
            img = np.array([img.T,img.T,img.T]) # C,H,W
        else: 
            # print(img.shape) # it is loaded in H,W,C
            img = img.transpose(2,1,0)

        # raise Exception('gg')
        img = resize_numpy_img_array(img, target_size=self.target_size, dims='CHW')
        # print(img.shape) # (1,960,960)
        # self.show_img(img[0].T)
        return img        
    
    def show_img(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()
        exit()

class PneuDataLoader(ImageProcessor):
    def __init__(self, DATA_DIR, split='train', resize=(960,960), n_debug=None, realtime_print=False):
        super(PneuDataLoader, self).__init__(resize=resize)


        self.split = split
        self.DATA_DIR = DATA_DIR # 'data/chest_xray'
        
        self.x = []
        self.y = []
        self.populate_data(n_debug=n_debug, realtime_print=realtime_print)

        assert(len(self.x)==len(self.y))
        self.data_size = len(self.y)

    def get_data(self, mode='shuffle', as_float_tensor=True, device=None, **kwargs):
        x,y0 = [],[]

        if mode=='shuffle':
            batch_size = kwargs['batch_size']
            draw = np.random.choice(range(self.data_size),size=batch_size)
            # print(draw)
            for i in draw:
                x.append(self.x[i])
                y0.append(self.y[i])
        elif mode=='by_index':
            i = kwargs['index']
            x.append(self.x[i])
            y0.append(self.y[i])
        else:
            raise RuntimeError('Invalid mode')

        x = np.array(x)
        y0 = np.array(y0)
        
        if as_float_tensor:
            x = torch.from_numpy(x).to(torch.float)
            y0 = torch.from_numpy(y0).to(torch.long)
        if device is not None:
            x = x.to(device=device)
            y0 = y0.to(device=device)
        return x, y0

    def populate_data(self, update_every=12, n_debug=0, realtime_print=False):
        """ Data
        train: NORMAL=1341, PNEUMONIA=3875, total=5216
        """
        IMG_NORMAL_DIR = os.path.join(self.DATA_DIR, self.split, 'NORMAL') 
        IMG_PNEUMONIA_DIR = os.path.join(self.DATA_DIR, self.split, 'PNEUMONIA') 
        
        def update_info(i,n, text=''):
            if (i+1)>=n or (i+1)%update_every==0:
                update_str = '%s: %s/%s'%(str(text),str(i+1),str(n))
                print('%-64s'%(str(update_str)),end='\r')

        def load_imgs_data(label, IMG_DIR, n_debug=0, text='', realtime_print=False):
            imgs = os.listdir(IMG_DIR)
            if n_debug>0: imgs = imgs[:n_debug]
            n = len(imgs)

            for i,x in enumerate(imgs):
                if realtime_print:
                    update_info(i,n,text=text)
                THIS_IMG_DIR = os.path.join(IMG_DIR, x)
                img = self.simple_img_preprocessing(pil_img=Image.open(THIS_IMG_DIR))
                self.x.append(img)
                self.y.append(label)
            print('Loading done from %s'%(str(IMG_DIR)))

        load_imgs_data(0, IMG_NORMAL_DIR, n_debug=n_debug, text='normal', realtime_print=realtime_print)
        load_imgs_data(1, IMG_PNEUMONIA_DIR, n_debug=n_debug, text='pneumonia', realtime_print=realtime_print)



