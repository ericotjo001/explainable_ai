import os, joblib
import numpy as np
import torch
from PIL import Image

from .dnn import device


def get_admission_th(L):
    return 0.5

def prep_data_and_dirs(dargs, modelname='model.pth'):
    print('prep_data_and_dirs...')

    folder_to_class_mapping = {f'{i}':i for i in range(10)}
    from src.settings.data_settings import CIFAR_SETTING
    rearrange_CIFAR_data(CIFAR_SETTING)

    ckptdir='checkpoint'
    dataname = 'CIFAR'
    CKPT_DIR = os.path.join(ckptdir, dataname) 
    MODEL_DIR = os.path.join(CKPT_DIR, modelname)
    TRAIN_RESULT_DIR = MODEL_DIR[:MODEL_DIR.find('.pth')]+'.train.result'
    TEST_RESULT_DIR = MODEL_DIR[:MODEL_DIR.find('.pth')]+'.test.result'

    HYPER_FOLDER_DIR = os.path.join(CKPT_DIR, 'hyper')

    CNN_MODEL_DIR = os.path.join(CKPT_DIR, 'cnn.pth')
    CNN_REPORT_DIR = os.path.join(CKPT_DIR, 'cnn.txt')
    
    os.makedirs(CKPT_DIR,exist_ok=True)

    DIRS = {
        'SOURCE_DATA_DIR': CIFAR_SETTING['SOURCE_DATA_DIR'],
        'DATA_DIR' : CIFAR_SETTING['DATA_DIR'],
        'TEST_DATA_DIR' : CIFAR_SETTING['TEST_DATA_DIR'],
    
        'CKPT_DIR': CKPT_DIR,
        'MODEL_DIR': MODEL_DIR,        
        'TRAIN_RESULT_DIR': TRAIN_RESULT_DIR,
        'TEST_RESULT_DIR':TEST_RESULT_DIR,

        'HYPER_FOLDER_DIR':HYPER_FOLDER_DIR,

        'CNN_MODEL_DIR': CNN_MODEL_DIR,
        'CNN_REPORT_DIR': CNN_REPORT_DIR,

    }
    return folder_to_class_mapping, DIRS

def rearrange_CIFAR_data(CIFAR_SETTING):
    if not os.path.exists(CIFAR_SETTING['DATA_DIR']):
        import torchvision
        print('preparing rearranged data...')

        from .cifar_data import CIFAR10Dataset
        cif = CIFAR10Dataset(train=True, root_dir=CIFAR_SETTING['SOURCE_DATA_DIR'])
        ciftest = CIFAR10Dataset(train=False, root_dir=CIFAR_SETTING['SOURCE_DATA_DIR'])

        trainset = cif.dataset
        testset = ciftest.dataset

        def rearrange(dataset, DIR, text='hello'):
            n = len(dataset)
            for i, dat in enumerate(dataset):
                img,y0 = dataset.__getitem__(i)

                class_dir = os.path.join(DIR,str(y0))
                os.makedirs(class_dir,exist_ok=True)
                img_dir = os.path.join(class_dir, '%s.png'%(str(i)))
                img.save(img_dir)

                if (i+1)%1000==0:
                    update_text = '%s/%s'%(str(i+1),str(n))
                    print('%-16s'%(str(update_text)), end='\r')
            print('\n%s done!'%(str(text)))
        rearrange(trainset, CIFAR_SETTING['DATA_DIR'], text='trainset')
        rearrange(testset, CIFAR_SETTING['TEST_DATA_DIR'], text='testset')
    else:
        print('using rearranged folder...')


def prep_deep_neural_network_and_data_loader(dargs, parser, BOOLS, DIRS, device=None):
    DO_DNN_TRAINING = False
    if dargs['DNN_TRAINING']:
        print('DNN training activated!')
        DO_DNN_TRAINING = True

    if not os.path.exists(DIRS['CNN_MODEL_DIR']):
        print('CNN to be trained because it does not yet exist!')
        DO_DNN_TRAINING = True

    if DO_DNN_TRAINING:
        from .dnn import dnn_pipeline
        dnn_pipeline(parser, DIRS)
        return None, None
    else:
        print('loading dnn from %s'%(str(DIRS['CNN_MODEL_DIR'])))
        dnn = torch.load(DIRS['CNN_MODEL_DIR'])
    dnn = dnn.to(device=device)
    dnn.eval()

    from .dnn import normalizeTransform

    def cifar_img_loader(data_dir):
        pil_img = Image.open(data_dir)
        img = np.asarray(pil_img)/255.
        # print(img.shape, np.max(img), np.min(img))
        img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
        img = img.to(device=device).to(torch.float)

        x = dnn( normalizeTransform(img) )
        x = x.clone().detach().cpu().numpy()
        x = x[0] # batch size=1, so we take the first item in the batch

        return x    
    return dnn, cifar_img_loader