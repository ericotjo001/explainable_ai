"""
"""

import os
import numpy as np
from PIL import Image

import torch
import torchvision.models as mod

from ..imagenet.imagenet_prep import prep_data_and_dirs, imagenet_reshape
from ..imagenet.dnn import normalizeTransform, device

N_CLASS = 1000

def get_admission_th(L):
    return 0.5

def prep_deep_neural_network_and_data_loader(DIRS, device=None):
    dnn = mod.resnet18(weights=mod.ResNet18_Weights.DEFAULT, progress=False)
    dnn = dnn.to(device=device)
    dnn.eval()
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

        x = np.mean(x.reshape(10,-1),axis=1) # a rather arbitrary way to reduce dimension
        return x
    return dnn, imagenet_img_loader

def prep_data_and_dirs(modelname='model.pth'):
    from src.settings.data_settings import IMAGENETV2_SETTING
    folder_to_class_mapping = {f'{i}':i for i in range(N_CLASS)}
    ckptdir='checkpoint'
    dataname = 'imagenetv2'

    CKPT_DIR = os.path.join(ckptdir, dataname) 
    MODEL_DIR = os.path.join(CKPT_DIR, modelname)
    HYPER_FOLDER_DIR = os.path.join(CKPT_DIR, 'hyper')
    DIRS = {
        'DATA_DIR': IMAGENETV2_SETTING['MFREQ_DATA_DIR'],
        'TEST_DATA_DIR': IMAGENETV2_SETTING['0.7FREQ_DATA_DIR'],

        'CKPT_DIR': CKPT_DIR,
        'MODEL_DIR': MODEL_DIR,        

        'HYPER_FOLDER_DIR':HYPER_FOLDER_DIR,
    }
    return folder_to_class_mapping, DIRS    

def imagenetv2_hyper_():
    """
    Experiment on data ordering as hyperparameter!
    """
    print('imagenetv2_hyper_')
    folder_to_class_mapping, DIRS = prep_data_and_dirs()
    dnn, imagenet_img_loader = prep_deep_neural_network_and_data_loader(DIRS, device=device)

    from datetime import datetime
    now = datetime.now()
    dated_showcase = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(DIRS['HYPER_FOLDER_DIR'],exist_ok=True)
    HYPER_RESULT_DIR = os.path.join(DIRS['HYPER_FOLDER_DIR'], 'hyper_%s.result'%(str(dated_showcase)))
    HYPER_BOXPLOT_DIR = os.path.join(DIRS['HYPER_FOLDER_DIR'], 'box_%s.png'%(str(dated_showcase)))
    
    from src.model.kabedonn import KABEDONN    
    settings = {'init_new':True,
            'folder_to_class_mapping': folder_to_class_mapping,
            'DATA_DIR': DIRS['DATA_DIR'],
            'n_class':len(folder_to_class_mapping),
            'kwidth': None, 
            'data_fetcher': imagenet_img_loader,
            'interpolator_settings': None,
            'activation_threshold': 0.999,
            'admission_threshold':get_admission_th,

        }

    FIRSTN = 10 # 10 images in each class
    fittingconfig={'print_final_info':False,'balance_test': True,
        'qconfig': {
            'mode': 'scrambledfirstn',
            'classes': range(N_CLASS),
            'firstn': [FIRSTN]*N_CLASS,
        }
    }

    from src.model.eval import evaluate_on_test_data
    eval_settings = {
        'DIRS':DIRS,
        'folder_to_class_mapping': folder_to_class_mapping,
        'data_fetcher': imagenet_img_loader,        
    }    


    n_trials = 3 # 8
    kwidths = [32] # [8, 16, 32, 64]
    labels = [str(k) for k in kwidths]
    labels.insert(0,'')

    accs = {k:[] for k in kwidths}
    for kwidth in kwidths:
        for i in range(n_trials):
            settings['kwidth'] = kwidth 
            print('==== kwidth:%s [%s] ===='%(str(kwidth),str(i)))
            net = KABEDONN(**settings)                    
            qlist = net.fit_data(config=fittingconfig,verbose=100)
            net.evaluate_and_finetune_on_train_data(qlist=qlist, verbose=0)

            TEST_EVAL_RESULT = evaluate_on_test_data(net, **eval_settings)
            TEST_EVAL_RESULT.print_accuracy()
            acc = TEST_EVAL_RESULT.correct/TEST_EVAL_RESULT.total
            # acc = np.random.normal(0,1,)
            accs[kwidth].append(acc)

    joblib.dump(accs, HYPER_RESULT_DIR)

    plt.figure()
    plt.boxplot([y for x,y in accs.items()], 
        flierprops={'marker':'.', 
        'markeredgecolor':'r',
        'markerfacecolor':'r', 
        'markersize':10,}
    )
    plt.xticks( range(len(kwidths)+1), labels)
    plt.xlim([0.5,None])
    plt.gca().set_xlabel('k')

    plt.savefig(HYPER_BOXPLOT_DIR)