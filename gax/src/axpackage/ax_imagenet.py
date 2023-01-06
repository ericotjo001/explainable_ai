from ..imagenet_utils import ImageNetValidation
from ..utils import *

from .ax import AugEplanation, compute_co_score, softmax

DEBUG_TOGGLES = {
    'val_iter': 0,
    'vis_iter': 0,
}

import warnings
warnings.filterwarnings('ignore')
# remove warning about captum removing forward, backward hooks

def manage_dirs(dargs):
    ROOT_DIR = os.getcwd() if dargs['ROOT_DIR'] is None else dargs['ROOT_DIR']
    CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
    os.makedirs(CKPT_DIR, exist_ok=True)

    PROJECT_DIR = os.path.join(CKPT_DIR, f"{dargs['data']}_{dargs['model']}")
    os.makedirs(PROJECT_DIR, exist_ok=True)

    # imagenet
    IMAGENET_AX_CLASS_ACC_DIR = os.path.join(PROJECT_DIR, 'ax_class_act.json')
    IMAGENET_CO_SCORE_DIR = os.path.join(PROJECT_DIR, f"co_score_{dargs['ax_method']}.scores")

    VIS_DIR = os.path.join(PROJECT_DIR,'visualization')
    os.makedirs(VIS_DIR, exist_ok=True)    

    DIRS = {
        'ROOT_DIR': ROOT_DIR,
        'CKPT_DIR': CKPT_DIR,
        'PROJECT_DIR': PROJECT_DIR,
        
        'IMAGENET_AX_CLASS_ACC_DIR':IMAGENET_AX_CLASS_ACC_DIR,
        'IMAGENET_CO_SCORE_DIR': IMAGENET_CO_SCORE_DIR,

        'VIS_DIR': VIS_DIR,
    }
    return DIRS

def imagenet_val(dargs, ax_method=None):
    assert (dargs['IMGNET_DATA_DIR'] is not None)
    assert (dargs['model'] is not None)

    DIRS = manage_dirs(dargs)

    n_class = 1000
    imgn = ImageNetValidation(MAIN_DATA_DIR=dargs['IMGNET_DATA_DIR'])

    from ..model import pretrained_model_selection_for_imagenet
    net = pretrained_model_selection_for_imagenet(model=dargs['model'])
    net.eval()

    from torchvision import transforms
    normalize = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    ax = AugEplanation(method=ax_method, model=dargs['model'])
    
    label = f"{dargs['data']}_{dargs['model']}_{str(dargs['ax_method'])}"
    
    result = {}
    if os.path.exists(DIRS['IMAGENET_AX_CLASS_ACC_DIR']):
        with open(DIRS['IMAGENET_AX_CLASS_ACC_DIR']) as f: 
            result = json.load(f)

    if ax_method is not None:             
        co_scores = []

    print('validating...')
    n_correct, n_total = 0, 50000
    with torch.no_grad():
        for i in range(50000):
            img, y0, label_text = imgn.get_val_data_by_index(i, as_pytorch_tensor=True)
            
            if img is None: continue # in case of problematic val data

            # print(img.shape) # torch.Size([1, 3, 375, 500])
            img = img.to(device=device)
            x = normalize(img)
            # print(img.shape) # torch.Size([1, 3, 256, 256])

            if ax_method is None:          
                y = net(x)
            else:
                y_baseline = net(x)

                attr = ax.get_attribution_by_method(net, x)
                attr = ax.normalize_attr(attr)
                y = net(x + attr)
            # print(y.shape) # torch.Size([1, 1000])

            y_pred = torch.argmax(y,dim=1)[0].item()
            # print(y_pred, y0) # 65 tensor([65])

            isCorrect = int(y_pred)==int(y0)
            if isCorrect:
                n_correct+=1

            if ax_method is not None: 
                co_score = compute_co_score(y_baseline, y, y0, n_class)
                co_score_softmax = compute_co_score(y_baseline, softmax(y), y0, n_class)
                co_score_info = (co_score, co_score_softmax, isCorrect)
                co_scores.append(co_score_info)

            if (i+1)%10==0 or (i+1)==50000:
                print('acc:%s/%s=%s'%(str(n_correct),str(i+1),str(n_correct/(i+1))),end='\r')    

            if DEBUG_TOGGLES['val_iter']:
                if i+1>=20: break


    result[label] = {'n_correct': n_correct, 'n_total': i+1, 'acc':n_correct/(i+1)}
    with open(DIRS['IMAGENET_AX_CLASS_ACC_DIR'], 'w') as json_file:
        json.dump(result, json_file, indent=4, sort_keys=True)

    if ax_method is not None: 
        joblib.dump({'co_scores':co_scores,'ax_method': ax_method}, DIRS['IMAGENET_CO_SCORE_DIR'])


def vis_co_score(dargs):
    print('vis_co_score')

    DIRS = manage_dirs(dargs)

    from .ax_vis import save_vis_dir, save_acc_table
    save_vis_dir(DIRS['PROJECT_DIR'], DIRS['VIS_DIR'], 
        DEBUG_TOGGLE_VIS_ITER=DEBUG_TOGGLES['vis_iter'])
    save_acc_table( dargs['data'], dargs['model'], 
        DIRS['IMAGENET_AX_CLASS_ACC_DIR'] , DIRS['VIS_DIR'])