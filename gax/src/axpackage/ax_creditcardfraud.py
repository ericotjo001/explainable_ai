from ..utils import *
from ..trainval import manage_dirs_creditcardfraud

from .ax import AugEplanation, compute_co_score, softmax

DEBUG_TOGGLES = {
    'val_iter': 0,
    'vis_iter': 0,
}
import warnings
warnings.filterwarnings('ignore')
# remove warning about captum removing forward, backward hooks


def ax_creditcardfraud_test(dargs, ax_method=None):
    print('ax_creditcardfraud_test')
    model = 'ccfFPA'
    DIRS = manage_dirs_creditcardfraud(dargs)

    n_class = 2
    from ..other_data import CreditcardFraudData
    testdataset = CreditcardFraudData(DIRS['REC_TEST_DATA_DIR'])   
    from torch.utils.data import DataLoader
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False) 

    from ..model import ccfFPA
    net = ccfFPA().to(device=device)
    ckpt = torch.load(DIRS['MODEL_DIR'])
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()

    ax = AugEplanation(method=ax_method, model=model)

    label = f"{dargs['data']}_{model}_{str(dargs['ax_method'])}"

    result = {}
    if os.path.exists(DIRS['CCF_AX_CLASS_ACC_DIR']):
        with open(DIRS['CCF_AX_CLASS_ACC_DIR']) as f:
            result = json.load(f)    

    if ax_method is not None:             
        co_scores = []

    n = len(testloader)
    n_correct = 0
    with torch.no_grad():
        for i, (x,y0) in enumerate(testloader):
            x = x.to(torch.float).to(device=device)
            y0 = y0.to(device=device)

            if ax_method is None:          
                y = net(x)
            else:
                y_baseline = net(x)

                attr = ax.get_attribution_by_method(net, x)
                attr = ax.normalize_attr(attr)
                y = net(x + attr)
            
            y_pred = torch.argmax(y,dim=1)[0].item()

            isCorrect = int(y_pred)==int(y0.item())
            if isCorrect:
                n_correct+=1

            if ax_method is not None: 
                co_score = compute_co_score(y_baseline, y, y0, n_class)
                co_score_softmax = compute_co_score(y_baseline, softmax(y), y0, n_class)
                co_score_info = (co_score, co_score_softmax, isCorrect)
                co_scores.append(co_score_info)


            if (i+1)%10==0 or (i+1)==n:
                print('%s/%s acc:%s/%s=%s'%(str(i+1),str(n),
                    str(n_correct),str(i+1),str(n_correct/(i+1))),end='\r')    

            if DEBUG_TOGGLES['val_iter']:
                if i+1>=20: break

    result[label] = {'n_correct': n_correct, 'n_total': i+1, 'acc':n_correct/(i+1)}
    with open(DIRS['CCF_AX_CLASS_ACC_DIR'], 'w') as json_file:
        json.dump(result, json_file, indent=4, sort_keys=True)

    if ax_method is not None: 
        joblib.dump({'co_scores':co_scores,'ax_method': ax_method}, DIRS['CCF_CO_SCORE_DIR'])

def ax_creditcardfraud_test_remaining(dargs):
    """
    Creditcard fraud dataset is very very imbalanced. We have so far
    used a few small subset of negative samples just to achieve some sort of balance
    with cp augmentation technique. 
    The "remaining" negative samples will be tested here, only for their predictive accuracy.
    """
    print('ax_creditcardfraud_test_remaining')
    model = 'ccfFPA'
    DIRS = manage_dirs_creditcardfraud(dargs)

    n_class = 2
    from ..other_data import CreditcardFraudData
    testdataset = CreditcardFraudData(DIRS['REC_TEST_REMAINING_DATA_DIR'])   
    from torch.utils.data import DataLoader
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False)  
    
    from ..model import ccfFPA
    net = ccfFPA().to(device=device)
    ckpt = torch.load(DIRS['MODEL_DIR'])
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()

    label = f"{dargs['data']}_{model}_remaining"

    result = {}
    if os.path.exists(DIRS['CCF_AX_CLASS_ACC_DIR']):
        with open(DIRS['CCF_AX_CLASS_ACC_DIR']) as f:
            result = json.load(f)    


    n = len(testloader)
    n_correct = 0
    with torch.no_grad():
        for i, (x,y0) in enumerate(testloader):
            x = x.to(torch.float).to(device=device)
            y0 = y0.to(device=device)

            y = net(x)
            y_pred = torch.argmax(y,dim=1)[0].item()

            isCorrect = int(y_pred)==int(y0.item())
            if isCorrect:
                n_correct+=1

            if (i+1)%10==0 or (i+1)==n:
                print('%s/%s acc:%s/%s=%s'%(str(i+1),str(n),
                    str(n_correct),str(i+1),str(n_correct/(i+1))),end='\r')    

            if DEBUG_TOGGLES['val_iter']:
                if i+1>=20: break
    result[label] = {'n_correct': n_correct, 'n_total': i+1, 'acc':n_correct/(i+1)}
    with open(DIRS['CCF_AX_CLASS_ACC_DIR'], 'w') as json_file:
        json.dump(result, json_file, indent=4, sort_keys=True)


def vis_co_score(dargs):
    print('vis_co_score')

    DIRS = manage_dirs_creditcardfraud(dargs)

    model = 'ccfFPA'
    from .ax_vis import save_vis_dir, save_acc_table
    save_vis_dir(DIRS['PROJECT_DIR'], DIRS['VIS_DIR'], 
        DEBUG_TOGGLE_VIS_ITER=DEBUG_TOGGLES['vis_iter'])
    save_acc_table( dargs['data'], model, 
        DIRS['CCF_AX_CLASS_ACC_DIR'] , DIRS['VIS_DIR'])