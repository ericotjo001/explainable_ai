# tested on torch==2.0.0.dev20221226+cu117

import os, json, joblib
import numpy as np

from xml.dom.minidom import parse
from PIL import Image
from skimage.transform import resize

from .imagenet_dict import LABEL_DIR
from .imagenet_dict2 import ID2LABELS

import torch
import torch.nn as nn


class ImageNetDirectQuery():
    """ Compared to ImageNetClassificationManager, this class directly, randomly query data.
    We don't bother creating subsets, just get whatever random data we get from our random values.
    """
    def __init__(self, MAIN_DATA_DIR=None, N_DEBUG=0):
        super(ImageNetDirectQuery, self).__init__()
        """
        MAIN_DATA_DIR:  dir to ILSVRC folder downloaded from kaggle
        """
        self.N_DEBUG=N_DEBUG
        
        self.MAIN_DATA_DIR = 'data' if MAIN_DATA_DIR is None else MAIN_DATA_DIR
        self.MAIN_TRAINING_DATA_DIR = os.path.join(self.MAIN_DATA_DIR,'Data','CLS-LOC')
        self.MAIN_DATA_BRANCH_DIR = os.path.join(self.MAIN_DATA_DIR,'Data','CLS-LOC', 'train')
        self.MAIN_DATA_TRAIN_CLS_LIST_DIR = os.path.join(self.MAIN_DATA_DIR,'ImageSets','CLS-LOC', 'train_cls.txt')        

        self.TRAINING_NAME_LIST = []
        self.TRAINING_LABEL_LIST = []
        self.FOLDER_ID_FOUND = []
        self.FOLDER_ID_NOT_FOUND = []
        self.load_training_list_with_txt_file()

    def load_training_list_with_txt_file(self):
        txt = open(self.MAIN_DATA_TRAIN_CLS_LIST_DIR)
        for i,x in enumerate(txt):
            if self.N_DEBUG>0:
                if i>self.N_DEBUG:
                    break
            folder_label,this_name = x.split('/')
            this_name = this_name.split(' ')[0] # this looks like n01440764_10048
            self.TRAINING_NAME_LIST.append(this_name)
            folder_label2 = this_name.split('_')[0]
            try:
                assert(folder_label==folder_label2) # no problem it seems
            except:
                print('label inconsistent?') 
            try:
                assert(folder_label2 in ID2LABELS) # problem here
                if not folder_label2 in self.FOLDER_ID_FOUND:
                    self.FOLDER_ID_FOUND.append(folder_label2)
            except:
                # print('folder label not found? %s'%(str(folder_label2)))
                if not folder_label2 in self.FOLDER_ID_NOT_FOUND :
                    self.FOLDER_ID_NOT_FOUND.append(folder_label2)
                continue

            y0, label_text = ID2LABELS[folder_label2]
            self.TRAINING_LABEL_LIST.append(y0)
        txt.close()
        print('len(self.FOLDER_ID_FOUND):',len(self.FOLDER_ID_FOUND))
        print('len(self.FOLDER_ID_NOT_FOUND):',len(self.FOLDER_ID_NOT_FOUND))
        print('self.FOLDER_ID_NOT_FOUND:',self.FOLDER_ID_NOT_FOUND)

    def get_one_training_sample_by_index(self, i, as_pytorch_tensor=True):
        this_name = self.TRAINING_NAME_LIST[i]
        folder_name = this_name.split('_')[0]
        img_dir = os.path.join(self.MAIN_DATA_BRANCH_DIR,folder_name,this_name + '.JPEG')
        pil_img = Image.open(img_dir)
        img = np.asarray(pil_img)/255.

        this_index, label_text = ID2LABELS[folder_name]

        if as_pytorch_tensor:
            s = img.shape
            if len(s)== 2:
                img = [img, img, img] # set channel to 3
            elif len(s) == 3:
                img = img.transpose(2,0,1)
            else:
                print(s)
                raise RuntimeError('What image is this???')
            img = torch.from_numpy(np.array([img])).to(torch.float)

        return img, this_index, label_text

    def get_pytorch_batch_samples(self, batch_size, input_size=(256,256), device=None):
        x, y0 = [], []
        for i in np.random.randint(0,len(self.TRAINING_NAME_LIST), size=(batch_size,)):
            img, this_index, label_text = self.get_one_training_sample_by_index(i, as_pytorch_tensor=False)
            s = img.shape
            if len(s)== 2:
                img = resize(img, input_size)
                img = [img, img, img] # set channel to 3
            elif len(s) == 3:
                img = resize(img, input_size + (3,))
                img = img.transpose(2,0,1)

            x.append(img)
            y0.append(this_index)

        x = torch.from_numpy(np.array(x)).to(torch.float).to(device=device)
        y0 = torch.from_numpy(np.array(y0)).to(torch.long).to(device=device)
        return x, y0


class ImageNetClassificationManager(object):
    def __init__(self, INDICES, PROJECT_DIR, split='train', MAIN_DATA_DIR=None):
        super(ImageNetClassificationManager, self).__init__()

        """
        This class is intended to extract subsets of the very large ImageNet. 
        Some parts may look redundant. See below, when classes for val and test datasets inherit this class,
            some properties are not used. This is because test and val datasets are not arranged in folders
            like the train datasets. Do not worry, just ignore them since they're harmless.
        
        PROJECT_DIR: dir to save intermediate list of data subset
        MAIN_DATA_DIR:  dir to ILSVRC folder downloaded from kaggle
        INDICES is a list of indices from 0,1,...,999 corresponding to the ImageNet classes
            for example INDICES=[1,2,5] means we are interested only to extract classes 1,2,5 for this data
        """
        self.PROJECT_DATA_DIR = os.path.join(PROJECT_DIR,'data_manager')
        os.makedirs(self.PROJECT_DATA_DIR, exist_ok=True)
        self.DATASET_DIR = os.path.join(self.PROJECT_DATA_DIR,'%s.subset'%(split))
        self.MAIN_DATA_DIR = 'data' if MAIN_DATA_DIR is None else MAIN_DATA_DIR
        self.MAIN_DATA_BRANCH_DIR = os.path.join(self.MAIN_DATA_DIR,'Data','CLS-LOC', split)
        self.MAIN_VAL_LABELS_DIR = os.path.join(self.MAIN_DATA_DIR,'Annotations','CLS-LOC', 'val')

        self.LABEL_DICT = LABEL_DIR
        # print(self.LABEL_DICT)
        self.INDICES = INDICES

    def get_subset_by_indices_for_classification(self, n_subset=None, split='train', create_new=True):
        # INDICES is a list that is a subset of [0,1,2,...,999]
        print('get_subset_by_indices_for_classification()')
        if not os.path.exists(self.DATASET_DIR) or create_new:
            print('creating new subset...')
            self.CURRENT_IMG_LIST = []
            self.CURRENT_LABEL_LIST = []
        else:
            print('loading subset...')
            self.CURRENT_IMG_LIST, self.CURRENT_LABEL_LIST = joblib.load(self.DATASET_DIR)
            return

        if split=='train':
            for this_index in self.INDICES:
                this_id = self.LABEL_DICT[str(this_index)]['id']
                folder_id = 'n'+this_id.split('-')[0]

                class_folder_dir = os.path.join(self.MAIN_DATA_BRANCH_DIR, folder_id)
                CLASS_IMG_LIST = os.listdir(class_folder_dir) # image name, e.g. n01440764_10026.JPEG
                
                if n_subset is None:
                    SELECTED_IMG_LIST = CLASS_IMG_LIST
                else:
                    CHOSEN_INDICES = np.random.randint(0,len(CLASS_IMG_LIST), size=n_subset)
                    SELECTED_IMG_LIST = [CLASS_IMG_LIST[x] for x in CHOSEN_INDICES]

                self.CURRENT_IMG_LIST = self.CURRENT_IMG_LIST + SELECTED_IMG_LIST # x
                self.CURRENT_LABEL_LIST = self.CURRENT_LABEL_LIST + [this_index for _ in range(len(SELECTED_IMG_LIST))] # y0

        elif split=='val':
            VAL_IMG_LIST = os.listdir(self.MAIN_DATA_BRANCH_DIR)

            if n_subset is None:
                SELECTED_IMG_LIST = VAL_IMG_LIST
            else:
                # CHOSEN_INDICES = np.random.randint(0,len(VAL_IMG_LIST), size=n_subset)
                CHOSEN_INDICES = range(n_subset)
                SELECTED_IMG_LIST = [VAL_IMG_LIST[x] for x in CHOSEN_INDICES]
                SELECTED_LABEL_LIST = []
                for i in CHOSEN_INDICES:
                    img_xml_name = SELECTED_IMG_LIST[i].split('.')[0] + '.xml'
                    xml_dir = os.path.join(self.MAIN_VAL_LABELS_DIR,img_xml_name)
                    # print(img_xml_name)
                    xml = parse(xml_dir)
                    this_node_list = xml.getElementsByTagName('name')
                    this_node_name = this_node_list[0].childNodes[0].nodeValue
                    this_index, label_text = ID2LABELS[this_node_name]
                    SELECTED_LABEL_LIST.append(this_index)
                self.CURRENT_IMG_LIST = self.CURRENT_IMG_LIST + SELECTED_IMG_LIST # x
                self.CURRENT_LABEL_LIST = self.CURRENT_LABEL_LIST + SELECTED_LABEL_LIST # y0
        else:
            raise RuntimeError('Not implemented')

        joblib.dump((self.CURRENT_IMG_LIST, self.CURRENT_LABEL_LIST), self.DATASET_DIR)

    def get_data_from_subset(self):
        # this is to use the data collected by
        # get_subset_by_indices_for_classification()
        raise RuntimeError('Not implemented')

class ImageNetValidation(ImageNetClassificationManager):
    def __init__(self, MAIN_DATA_DIR=None):
        super(ImageNetValidation, self).__init__(None,'',split='val', MAIN_DATA_DIR=MAIN_DATA_DIR)
        self.VAL_IMG_LIST = os.listdir(self.MAIN_DATA_BRANCH_DIR)

    def get_val_data_by_index(self, i, as_pytorch_tensor=True):
        assert(1<=i+1<=50000)
        img_xml_name = self.VAL_IMG_LIST[i].split('.')[0] + '.xml'
        img_name = self.VAL_IMG_LIST[i].split('.')[0] + '.JPEG'

        xml_dir = os.path.join(self.MAIN_VAL_LABELS_DIR,img_xml_name)
        img_dir = os.path.join(self.MAIN_DATA_BRANCH_DIR,img_name)

        xml = parse(xml_dir)
        this_node_list = xml.getElementsByTagName('name')
        this_node_name = this_node_list[0].childNodes[0].nodeValue
        try:
            this_index, label_text = ID2LABELS[this_node_name]
        except:
            return None, None, None

        pil_img = Image.open(img_dir)
        img = np.asarray(pil_img)/255.

        # print(img.shape)
        if as_pytorch_tensor:
            s = img.shape
            if len(s)== 2:
                img = [img, img, img] # set channel to 3
            elif len(s) == 3:
                img = img.transpose(2,0,1)
            else:
                print(s)
                raise RuntimeError('What image is this???')
            img = torch.from_numpy(np.array([img])).to(torch.float)
            this_index = torch.tensor([this_index]).to(torch.long)

        return img, this_index, label_text

class ImageNetTest(ImageNetClassificationManager):
    """Note: the test dataset has no annotation"""
    def __init__(self, MAIN_DATA_DIR=None):
        super(ImageNetTest, self).__init__(None,'',split='test', MAIN_DATA_DIR=MAIN_DATA_DIR)
        self.TEST_IMG_LIST = os.listdir(self.MAIN_DATA_BRANCH_DIR)

    def get_test_data_by_index(self,i, as_pytorch_tensor=True):
        assert(1<=i+1<=100000)
        img_name = self.TEST_IMG_LIST[i].split('.')[0] + '.JPEG'
        img_dir = os.path.join(self.MAIN_DATA_BRANCH_DIR,img_name)
        pil_img = Image.open(img_dir)
        img = np.asarray(pil_img)/255.

        # print(img.shape)
        if as_pytorch_tensor:
            s = img.shape
            if len(s)== 2:
                img = [img, img, img] # set channel to 3
            elif len(s) == 3:
                img = img.transpose(2,0,1)
            else:
                print(s)
                raise RuntimeError('What image is this???')
            img = torch.from_numpy(np.array([img])).to(torch.float)

        return img