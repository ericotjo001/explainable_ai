from .utils import *

DEBUG_TOGGLES = {
    'trainval_iter': 0,
}

"""
CONTENT PAGE:
1. CHEST X RAY COVID
2. CREDIT CARD FRAUD
3. DRY BEAN
"""

####################################
#       CHEST X RAY COVID
####################################

def manage_dirs_chestxray_covid(dargs):
    ROOT_DIR = os.getcwd()
    CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
    os.makedirs(CKPT_DIR, exist_ok=True)

    PROJECT_DIR = os.path.join(CKPT_DIR, f"chestxray_covid-{dargs['label_name']}")
    os.makedirs(PROJECT_DIR, exist_ok=True)

    DATA_SPLIT_CACHE = os.path.join(PROJECT_DIR, 'data_split_cache')
    DATA_VIS_DIR = os.path.join(PROJECT_DIR, 'datavis')
    os.makedirs(DATA_VIS_DIR, exist_ok=True)

    MODEL_DIR = os.path.join(PROJECT_DIR, 'CXCMultiSPA.model')
    TRAINVAL_STATUS_DIR = os.path.join(PROJECT_DIR, 'trainval_status.json')

    CHEST_XRAY_COVID_AX_CLASS_ACC_DIR = os.path.join(PROJECT_DIR, 'ax_class_act.json')
    CHEST_XRAY_COVID_CO_SCORE_DIR = os.path.join(PROJECT_DIR, f"co_score_{dargs['ax_method']}.scores")

    VIS_DIR = os.path.join(PROJECT_DIR,'visualization')
    os.makedirs(VIS_DIR, exist_ok=True)  

    DIRS = {
        'ROOT_DIR': ROOT_DIR,
        'CKPT_DIR': CKPT_DIR,
        'PROJECT_DIR': PROJECT_DIR,
        'DATA_VIS_DIR': DATA_VIS_DIR,

        'CHEST_XRAY_COVID_DATA_DIR': dargs['CHEST_XRAY_COVID_DATA_DIR'],
        'DATA_SPLIT_CACHE':DATA_SPLIT_CACHE,
        'MODEL_DIR': MODEL_DIR, 
        'TRAINVAL_STATUS_DIR': TRAINVAL_STATUS_DIR,

        'CHEST_XRAY_COVID_AX_CLASS_ACC_DIR': CHEST_XRAY_COVID_AX_CLASS_ACC_DIR,
        'CHEST_XRAY_COVID_CO_SCORE_DIR': CHEST_XRAY_COVID_CO_SCORE_DIR,

        'VIS_DIR': VIS_DIR,
    }
    return DIRS

def chestxray_covid_visdata(dargs):
    print('chestxray_covid_trainval')

    DIRS = manage_dirs_chestxray_covid(dargs)

    SPLITS=['train', 'val', 'test']
    datavis = {split:[] for split in SPLITS}

    from .chestxray_data import chestXRayCovid
    for split in SPLITS:
        traindata = chestXRayCovid(DIRS['CHEST_XRAY_COVID_DATA_DIR'],
            DIRS['DATA_SPLIT_CACHE'],
            split=split,
            img_size=(256,256))

        for i,(imgname, y0) in enumerate(traindata.data_list[split]):
            img_dir = os.path.join(DIRS['CHEST_XRAY_COVID_DATA_DIR'],
                traindata.LABELS[y0], 'images', imgname)

            pil_img = Image.open(img_dir)
            img = np.array(pil_img)

            datavis[split].append([np.min(img), np.max(img)])

    plt.figure()
    for split in SPLITS:
        this_split = np.array(datavis[split])
        plt.scatter(this_split[:,0], this_split[:,1], label=split, alpha=0.2)
    plt.legend()
    plt.gca().set_xlabel('min')
    plt.gca().set_ylabel('max')
    plt.savefig(os.path.join(DIRS['DATA_VIS_DIR'],'maxmin.png')  )
    print('saving fig to %s'%(str(DIRS['DATA_VIS_DIR'])))



################## Training and Validation ##################

def get_transform(transformtype='one_channel'):
    from torchvision import transforms
    if transformtype=='one_channel':
        normalize = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.Normalize(mean=[0.456, ],std=[0.224,]),
        ])
    else:
        raise NotImplementedError()
    return normalize

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def chestxray_covid_trainval(dargs):
    print('chestxray_covid_trainval')

    DIRS = manage_dirs_chestxray_covid(dargs)

    from .chestxray_data import chestXRayCovid
    traindata = chestXRayCovid(DIRS['CHEST_XRAY_COVID_DATA_DIR'],
        DIRS['DATA_SPLIT_CACHE'],split='train',img_size=(256,256))
    valdata = chestXRayCovid(DIRS['CHEST_XRAY_COVID_DATA_DIR'],
        DIRS['DATA_SPLIT_CACHE'],split='val',img_size=(256,256))

    from torch.utils.data import DataLoader
    trainloader = DataLoader(traindata, batch_size=dargs['batch_size'], shuffle=True)
    valloader = DataLoader(valdata, batch_size=dargs['batch_size'], shuffle=True)

    normalize = get_transform(transformtype='one_channel')

    from .model import CXCMultiSPA
    net = CXCMultiSPA().to(device=device)
    n_params = count_parameters(net)

    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5,0.999))
    criterion = nn.CrossEntropyLoss()

    import time
    start = time.time()

    epoch  = dargs['n_epochs']
    niter = len(trainloader)
    STATUS = {'val_acc':-1., 'best_val_acc':-1.}
    for e in range(epoch):
        net.train()
        for i, (x,y0) in enumerate(trainloader):
            net.zero_grad()

            x = x.to(torch.float).to(device=device)
            y0 = y0.to(device=device)
            # print(x.shape, y0)
            # torch.Size([16, 1, 299, 299]) tensor([0, 1, 0, ..., 0, 2], device='cuda:0')

            y = net(normalize(x))

            loss = criterion(y,y0)
            loss.backward()
            optimizer.step()

            if (i+1)==niter or (i+1)%10==0:
                update_text = f'epoch {e}: {i+1}/{niter} | val acc:{round(STATUS["val_acc"],3)}'
                print('%-64s'%(update_text), end='\r')

                if DEBUG_TOGGLES['trainval_iter']: break

        VAL_STATUS = chestxray_covid_validate(net, valloader)
        STATUS["val_acc"] = VAL_STATUS['val_acc']
        if VAL_STATUS['val_acc'] > STATUS['best_val_acc']:
            torch.save({'model_state_dict': net.state_dict()}, DIRS['MODEL_DIR'])            
            STATUS.update({
                '_args': dargs,
                'best_val_acc': VAL_STATUS['val_acc'], 
                'n_params': n_params, 
                })
            with open(DIRS['TRAINVAL_STATUS_DIR'], 'w') as json_file:
                json.dump(STATUS, json_file, indent=4, sort_keys=True)

            if VAL_STATUS['val_acc']> dargs['VAL_TARGET']: 
                print(f"\nVALIDATION TARGET {dargs['VAL_TARGET']} reached at epoch {e}.")
                break

    end = time.time()
    elapsed = end - start
    print(' time taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), 
        str(round(elapsed/60.,1)), 
        str(round(elapsed/3600.,1))))

    print('\ntraining ended')
            

def chestxray_covid_validate(net, valloader):
    net.eval()
    normalize = get_transform(transformtype='one_channel')

    n_correct, n_total = 0,0
    for i, (x,y0) in enumerate(valloader):
        x = x.to(torch.float).to(device=device)
        y0 = y0.to(device=device)
        y = net(normalize(x))

        y_pred = torch.argmax(y,dim=1).detach().cpu().numpy()
        y0 = y0.clone().detach().cpu().numpy()
        
        # print(y_pred, y0)
        # [0 0 0 0 2 2 0 0 0 0 2 0 0 0 0 2] [0 3 0 0 1 3 1 3 1 0 2 2 0 2 2 2]
        # print(y_pred==y0)
        #[ True False  True  True False False False False False  True  True False True False False  True]

        n_correct += sum(y_pred==y0)
        n_total += len(y0)

        if DEBUG_TOGGLES['trainval_iter']:
            if i>10: break

    val_acc = n_correct/n_total
    
    VAL_STATUS = {
        'val_acc': val_acc,
    }
    return VAL_STATUS



####################################
#       CREDIT CARD FRAUD
####################################

def manage_dirs_creditcardfraud(dargs):
    ROOT_DIR = os.getcwd()
    CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
    os.makedirs(CKPT_DIR, exist_ok=True)

    PROJECT_DIR = os.path.join(CKPT_DIR, f"creditcardfraud-{dargs['label_name']}")
    os.makedirs(PROJECT_DIR, exist_ok=True)

    CREDIT_FRAUD_DATA_DIR = dargs['CREDIT_FRAUD_DATA_DIR']

    CCF_RECONSTRUCT_FOLDER_DIR = os.path.join(CKPT_DIR,'ccf-reconstructed-data')
    os.makedirs(CCF_RECONSTRUCT_FOLDER_DIR, exist_ok=True)
    REC_TRAIN_DATA_DIR = os.path.join(CCF_RECONSTRUCT_FOLDER_DIR, 'rec_train.csv')
    REC_VAL_DATA_DIR = os.path.join(CCF_RECONSTRUCT_FOLDER_DIR, 'rec_val.csv')
    REC_TEST_DATA_DIR = os.path.join(CCF_RECONSTRUCT_FOLDER_DIR, 'rec_test.csv')
    REC_TEST_REMAINING_DATA_DIR = os.path.join(CCF_RECONSTRUCT_FOLDER_DIR, 'rec_test_REM.csv')

    MODEL_DIR = os.path.join(PROJECT_DIR, 'ccfFPA.model')
    TRAINVAL_STATUS_DIR = os.path.join(PROJECT_DIR, 'trainval_status.json')

    CCF_AX_CLASS_ACC_DIR = os.path.join(PROJECT_DIR, 'ax_class_act.json')
    CCF_CO_SCORE_DIR = os.path.join(PROJECT_DIR, f"co_score_{dargs['ax_method']}.scores")

    VIS_DIR = os.path.join(PROJECT_DIR,'visualization')
    os.makedirs(VIS_DIR, exist_ok=True)    

    DIRS = {
        'ROOT_DIR': ROOT_DIR,
        'CKPT_DIR': CKPT_DIR,    
        'PROJECT_DIR': PROJECT_DIR,

        'CREDIT_FRAUD_DATA_DIR': CREDIT_FRAUD_DATA_DIR,

        'CCF_RECONSTRUCT_FOLDER_DIR': CCF_RECONSTRUCT_FOLDER_DIR,
        'REC_TRAIN_DATA_DIR': REC_TRAIN_DATA_DIR,
        'REC_VAL_DATA_DIR': REC_VAL_DATA_DIR,
        'REC_TEST_DATA_DIR': REC_TEST_DATA_DIR,
        'REC_TEST_REMAINING_DATA_DIR':REC_TEST_REMAINING_DATA_DIR,

        'MODEL_DIR': MODEL_DIR, 
        'TRAINVAL_STATUS_DIR': TRAINVAL_STATUS_DIR,

        'CCF_AX_CLASS_ACC_DIR': CCF_AX_CLASS_ACC_DIR,
        'CCF_CO_SCORE_DIR': CCF_CO_SCORE_DIR,

        'VIS_DIR': VIS_DIR,
    }
    return DIRS

def creditcardfraud_vis_data(dargs):
    DIRS = manage_dirs_creditcardfraud(dargs)

    ccf_df = pd.read_csv(DIRS['CREDIT_FRAUD_DATA_DIR'])

    print(ccf_df.columns)
    # Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    #  'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    #  'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
    #  'Class'],
    # dtype='object')
    
    print('Class:',set(ccf_df['Class'])) # Class: {0, 1}
    print('n class 0:', len(ccf_df[ccf_df['Class']==0])) # n class 0: 284315
    print('n class 1:', len(ccf_df[ccf_df['Class']==1])) # n class 1: 492

    plt.figure(figsize=(12,8))
    plt.gcf().add_subplot(2,2,1)
    plt.gca().hist(ccf_df['Time'], alpha=0.3, edgecolor='black', linewidth=1.2)
    plt.gca().set_xlabel('time')

    plt.gcf().add_subplot(2,2,2)
    plt.gca().boxplot([ ccf_df[f'Amount'] ],
        flierprops={'marker':'.', 'markeredgecolor':(1,0,0,0.2),
        'markerfacecolor':(1,0,0,0.2), 'markersize':1,})
    plt.gca().set_xlabel('Amount boxplot')


    plt.gcf().add_subplot(2,2,3)
    plt.gca().boxplot([ ccf_df[f'V{i}'] for i in range(1,1+14)],
        flierprops={'marker':'.', 'markeredgecolor':(1,0,0,0.2),
        'markerfacecolor':(1,0,0,0.2), 'markersize':1,})
    plt.gca().set_xlabel('PCA V1-14 boxplot')

    plt.gcf().add_subplot(2,2,4)
    plt.gca().boxplot([ ccf_df[f'V{i}'] for i in range(15,15+14)],
        flierprops={'marker':'.', 'markeredgecolor':(1,0,0,0.2),
        'markerfacecolor':(1,0,0,0.2), 'markersize':1,})
    plt.gca().set_xlabel('PCA V15-28 boxplot')

    plt.savefig(os.path.join(DIRS['CCF_RECONSTRUCT_FOLDER_DIR'], 'original_data_dist.png'))
    plt.close()

    for i in range(1,1+28):
        plt.figure()
        plt.hist(ccf_df[f'V{i}'] , alpha=0.3, edgecolor='black', linewidth=1.2,
            bins=np.arange(-50, 50, 1))
        plt.savefig(os.path.join(DIRS['CCF_RECONSTRUCT_FOLDER_DIR'], f'PCA-{i}.png'))
        plt.close()

# We want to reconstruct the data to handle data imbalance
def creditcardfraud_reconstruct_data(dargs):
    print('reconstructing data...')
    DIRS = manage_dirs_creditcardfraud(dargs)
    ccf_df = pd.read_csv(dargs['CREDIT_FRAUD_DATA_DIR'])

    df_negative = ccf_df[ccf_df['Class']==0] # size 284315
    df_positive = ccf_df[ccf_df['Class']==1] # size 492
    # print(len(df_negative), len(df_positive)) # 284315 492

    COLUMNS = [f"V{i}" for i in range(1,1+28)] + ['Class']
    df_positive = df_positive.loc[:,COLUMNS].reset_index()

    from .cp_augmentation import cp_augmentation_type_dfc
    # df_aug = cp_augmentation_type_dfc(df, df_ref, cross_factor=5, dev=0.95)

    # let's construct train data
    n1,n2,n3 = dargs['n_split_negative'] 
    
    df_train = df_negative.loc[:n1, COLUMNS].reset_index()
    # print(df_train.head()) # each row is like [v1, v2, ..., v28, label], label like 0 or 1
    df_train_aug = cp_augmentation_type_dfc(df_positive, df_train, cross_factor=5, dev=0.95)
    # print(df_train.shape, df_train_aug.shape) # (2499, 30) (2460, 30)
    # print(pd.concat([df_train, df_train_aug],join="inner", ignore_index=True).shape)  # (4959, 30)
    pd.concat([df_train, df_train_aug],join="inner").drop(columns=['index']).to_csv(DIRS['REC_TRAIN_DATA_DIR'], index=False)

    df_val = df_negative.loc[n1:(n1+n2), COLUMNS].reset_index()
    # print(df_val.head())
    df_val_aug = cp_augmentation_type_dfc(df_positive, df_val, cross_factor=5, dev=0.95)
    # print(df_val.shape, df_val_aug.shape) # (2500, 30) (2460, 30)
    # print(pd.concat([df_val, df_val_aug],join="inner", ignore_index=True).shape) # (4960, 30)
    pd.concat([df_val, df_val_aug],join="inner", ignore_index=True).drop(columns=['index']).to_csv(DIRS['REC_VAL_DATA_DIR'], index=False)

    df_test = df_negative.loc[(n1+n2):(n1+n2+n3), COLUMNS].reset_index()
    pd.concat([df_test, df_positive],join="inner", ignore_index=True).drop(columns=['index']).to_csv(DIRS['REC_TEST_DATA_DIR'], index=False)
    # print(pd.concat([df_test, df_positive],join="inner", ignore_index=True).drop(columns=['index']).shape) # (2971, 29)

    df_test_remaining = df_negative.loc[(n1+n2+n3):, COLUMNS].reset_index().drop(columns=['index'])
    # print(df_test_remaining.shape) # (276840, 29)
    df_test_remaining.to_csv(DIRS['REC_TEST_REMAINING_DATA_DIR'], index=False)


def creditcardfraud_trainval(dargs):
    print('creditcardfraud_trainval...')

    DIRS = manage_dirs_creditcardfraud(dargs)

    from .other_data import CreditcardFraudData
    traindataset = CreditcardFraudData(DIRS['REC_TRAIN_DATA_DIR'])
    valdataset = CreditcardFraudData(DIRS['REC_VAL_DATA_DIR'])

    from torch.utils.data import DataLoader
    trainloader = DataLoader(traindataset, batch_size=dargs['batch_size'], shuffle=True)
    valloader = DataLoader(valdataset, batch_size=dargs['batch_size'], shuffle=True)

    from .model import ccfFPA
    net = ccfFPA().to(device=device)
    n_params = count_parameters(net)

    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5,0.999))
    criterion = nn.CrossEntropyLoss()

    import time
    start = time.time()

    epoch = dargs['n_epochs']
    niter = len(trainloader)
    STATUS = {'val_acc':-1., 'best_val_acc':-1.}
    for e in range(epoch):
        net.train()
        for i, (x,y0) in enumerate(trainloader):
            net.zero_grad()

            x = x.to(torch.float).to(device=device)
            y0 = y0.to(device=device)
            # print(x.shape, y0) # torch.Size([16, 28]) tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1], device='cuda:0')

            y = net(x)

            loss = criterion(y,y0)
            loss.backward()
            optimizer.step()            

            if (i+1)==niter or (i+1)%10==0:
                update_text = f'epoch {e}: {i+1}/{niter} | val acc:{round(STATUS["val_acc"],3)}'
                print('%-64s'%(update_text), end='\r')

                if DEBUG_TOGGLES['trainval_iter']: break

        VAL_STATUS = creditcardfraud_validate(net, valloader)
        STATUS["val_acc"] = VAL_STATUS['val_acc']
        if VAL_STATUS['val_acc'] > STATUS['best_val_acc']:
            torch.save({'model_state_dict': net.state_dict()}, DIRS['MODEL_DIR'])            
            STATUS.update({
                '_args': dargs,
                'best_val_acc': VAL_STATUS['val_acc'], 
                'n_params': n_params, 
                })
            with open(DIRS['TRAINVAL_STATUS_DIR'], 'w') as json_file:
                json.dump(STATUS, json_file, indent=4, sort_keys=True)

            if VAL_STATUS['val_acc']> dargs['VAL_TARGET']: 
                print(f"\nVALIDATION TARGET {dargs['VAL_TARGET']} reached at epoch {e}.")
                break

    end = time.time()
    elapsed = end - start
    print(' time taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), 
        str(round(elapsed/60.,1)), 
        str(round(elapsed/3600.,1))))

    print('\ntraining ended')
    print('fraction_positive:',VAL_STATUS['fraction_positive'], '(should be about 0.5)')


def creditcardfraud_validate(net, valloader):
    net.eval()

    n_correct, n_total = 0,0
    n_positive = 0
    for i, (x,y0) in enumerate(valloader):
        x = x.to(torch.float).to(device=device)
        y0 = y0.to(device=device)
        y = net(x)

        y_pred = torch.argmax(y,dim=1).detach().cpu().numpy()
        y0 = y0.clone().detach().cpu().numpy()
        
        # print(y_pred, y0)
        # [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] [0 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1]
        # print(y_pred==y0)
        # [False False  True  True False  True  True False  True False False  True

        n_positive += np.sum(y0==1)
        n_correct += sum(y_pred==y0)
        n_total += len(y0)

        if DEBUG_TOGGLES['trainval_iter']:
            if i>10: break

    val_acc = n_correct/n_total
    
    VAL_STATUS = {
        'val_acc': val_acc,
        'fraction_positive': n_positive/n_total, # just to check augmented data is not imbalanced
    }
    return VAL_STATUS





####################################
#            DRY BEAN
####################################

def manage_dirs_drybean(dargs):
    ROOT_DIR = os.getcwd()
    CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
    os.makedirs(CKPT_DIR, exist_ok=True)

    PROJECT_DIR = os.path.join(CKPT_DIR, f"drybean-{dargs['label_name']}")
    os.makedirs(PROJECT_DIR, exist_ok=True)

    DRYBEAN_DATA_DIR = dargs['DRYBEAN_DATA_DIR']

    DRYBEAN_DATA_OBS_FOLDER_DIR = os.path.join(CKPT_DIR,'db-data-obs')
    os.makedirs(DRYBEAN_DATA_OBS_FOLDER_DIR, exist_ok=True)
    DRYBEAN_NORMALIZATION_DIR = os.path.join(DRYBEAN_DATA_OBS_FOLDER_DIR, 'normalization.json')
    REC_TRAIN_DATA_DIR = os.path.join(DRYBEAN_DATA_OBS_FOLDER_DIR, 'rec_train.csv')
    REC_VAL_DATA_DIR = os.path.join(DRYBEAN_DATA_OBS_FOLDER_DIR, 'rec_val.csv')
    REC_TEST_DATA_DIR = os.path.join(DRYBEAN_DATA_OBS_FOLDER_DIR, 'rec_test.csv')
    REC_TEST_REMAINING_DATA_DIR = os.path.join(DRYBEAN_DATA_OBS_FOLDER_DIR, 'rec_test_REM.csv')

    MODEL_DIR = os.path.join(PROJECT_DIR, 'dbFPA.model')
    TRAINVAL_STATUS_DIR = os.path.join(PROJECT_DIR, 'trainval_status.json')

    DRYBEAN_AX_CLASS_ACC_DIR = os.path.join(PROJECT_DIR, 'ax_class_act.json')
    DRYBEAN_CO_SCORE_DIR = os.path.join(PROJECT_DIR, f"co_score_{dargs['ax_method']}.scores")

    VIS_DIR = os.path.join(PROJECT_DIR,'visualization')
    os.makedirs(VIS_DIR, exist_ok=True)    

    DIRS = {
        'ROOT_DIR': ROOT_DIR,
        'CKPT_DIR': CKPT_DIR,    
        'PROJECT_DIR': PROJECT_DIR,

        'DRYBEAN_DATA_DIR': DRYBEAN_DATA_DIR,

        'DRYBEAN_DATA_OBS_FOLDER_DIR': DRYBEAN_DATA_OBS_FOLDER_DIR,
        'DRYBEAN_NORMALIZATION_DIR': DRYBEAN_NORMALIZATION_DIR,
        'REC_TRAIN_DATA_DIR': REC_TRAIN_DATA_DIR,
        'REC_VAL_DATA_DIR': REC_VAL_DATA_DIR,
        'REC_TEST_DATA_DIR': REC_TEST_DATA_DIR,
        'REC_TEST_REMAINING_DATA_DIR':REC_TEST_REMAINING_DATA_DIR,

        'MODEL_DIR': MODEL_DIR, 
        'TRAINVAL_STATUS_DIR': TRAINVAL_STATUS_DIR,

        'DRYBEAN_AX_CLASS_ACC_DIR': DRYBEAN_AX_CLASS_ACC_DIR,
        'DRYBEAN_CO_SCORE_DIR': DRYBEAN_CO_SCORE_DIR,

        'VIS_DIR': VIS_DIR,
    }
    return DIRS

def drybean_vis_data(dargs):
    print('drybean_vis_data...')

    DIRS = manage_dirs_drybean(dargs)

    db_df = pd.read_excel(DIRS['DRYBEAN_DATA_DIR'], index_col=None)
    print(db_df.columns)
    # Index(['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    #  'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent',
    #  'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
    #  'ShapeFactor3', 'ShapeFactor4', 'Class'],
    # dtype='object')

    normalization = {
        'mean': [], 'std': []
    }
    FEATURES = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
        'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent',
        'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
        'ShapeFactor3', 'ShapeFactor4']

    for x in FEATURES:
        normalization['mean'].append(np.mean(db_df[x]))
        normalization['std'].append(np.var(db_df[x])**0.5)

    for i,x in enumerate(FEATURES):
        plt.figure(figsize=(12,5))
        plt.gcf().add_subplot(1,2,1)
        plt.gca().hist(db_df[x] , alpha=0.3, edgecolor='black', linewidth=1.2,)
        plt.gca().set_xlabel(x)
        plt.gcf().add_subplot(1,2,2)
        plt.gca().hist( (np.array(db_df[x]) - normalization['mean'][i])/normalization['std'][i] , alpha=0.3, edgecolor='black', linewidth=1.2,)
        plt.tight_layout()
        plt.savefig(os.path.join(DIRS['DRYBEAN_DATA_OBS_FOLDER_DIR'], f'feature-{x}.png'))
        plt.close()

    with open(DIRS['DRYBEAN_NORMALIZATION_DIR'], 'w') as json_file:
        json.dump(normalization, json_file, indent=4, sort_keys=True)

    print(set(db_df['Class']))
    # {'BOMBAY', 'SEKER', 'BARBUNYA', 'DERMASON', 'CALI', 'HOROZ', 'SIRA'}

    class_count = {}
    # print(db_df.shape) # (13611, 17)
    for c in ['BOMBAY', 'SEKER', 'BARBUNYA', 'DERMASON', 'CALI', 'HOROZ', 'SIRA']: 
        class_count[c] = len(db_df[db_df['Class']==c])
    print(class_count)
    # {'BOMBAY': 522, 'SEKER': 2027, 'BARBUNYA': 1322, 'DERMASON': 3546, 
    #   'CALI': 1630, 'HOROZ': 1928, 'SIRA': 2636}

def drybean_reconstruct_data(dargs):
    print('drybean_reconstruct_data...')

    DIRS = manage_dirs_drybean(dargs)
    db_df = pd.read_excel(DIRS['DRYBEAN_DATA_DIR'], index_col=None)
    CLASSES = ['BOMBAY', 'SEKER', 'BARBUNYA', 'DERMASON', 'CALI', 'HOROZ', 'SIRA']

    TRAIN, VAL, TEST = [], [], []
    for c in CLASSES: 
        subdf = db_df[db_df['Class']==c].reset_index() # this at index to the first column
        TRAIN.append(subdf[subdf['index']%3==0])
        VAL.append(subdf[subdf['index']%3==1])
        TEST.append(subdf[subdf['index']%3==2])

    pd.concat(TRAIN).drop(columns='index').to_csv(DIRS['REC_TRAIN_DATA_DIR'], index=False)
    pd.concat(VAL).drop(columns='index').to_csv(DIRS['REC_VAL_DATA_DIR'], index=False)
    pd.concat(TEST).drop(columns='index').to_csv(DIRS['REC_TEST_DATA_DIR'], index=False)

    # let's double check
    for x in ['TRAIN', 'VAL', 'TEST']:
        print(x)
        df_temp = pd.read_csv(DIRS[f'REC_{x}_DATA_DIR'] ,float_precision='high', index_col=False)
        for c in CLASSES:
            print(f"  {c} : {len(df_temp[df_temp['Class']==c])}")
        print('  total:', len(df_temp))
    """
    TRAIN
      BOMBAY : 174
      SEKER : 676
      BARBUNYA : 441
      DERMASON : 1182
      CALI : 543
      HOROZ : 643
      SIRA : 878
      total: 4537
    VAL
      BOMBAY : 174
      SEKER : 676
      BARBUNYA : 440
      DERMASON : 1182
      CALI : 544
      HOROZ : 642
      SIRA : 879
      total: 4537
    TEST
      BOMBAY : 174
      SEKER : 675
      BARBUNYA : 441
      DERMASON : 1182
      CALI : 543
      HOROZ : 643
      SIRA : 879
      total: 4537    
    """

def drybean_trainval(dargs):
    print('drybean_trainval...')

    DIRS = manage_dirs_drybean(dargs)

    from .other_data import DryBeanDataset
    traindataset = DryBeanDataset(DIRS['REC_TRAIN_DATA_DIR'], DIRS['DRYBEAN_NORMALIZATION_DIR'])
    valdataset = DryBeanDataset(DIRS['REC_VAL_DATA_DIR'], DIRS['DRYBEAN_NORMALIZATION_DIR'])

    from torch.utils.data import DataLoader
    trainloader = DataLoader(traindataset, batch_size=dargs['batch_size'], shuffle=True)
    valloader = DataLoader(valdataset, batch_size=dargs['batch_size'], shuffle=True)

    from .model import drybeanFPA
    net = drybeanFPA().to(device=device)
    n_params = count_parameters(net) # 

    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5,0.999))
    criterion = nn.CrossEntropyLoss()

    import time
    start = time.time()

    epoch = dargs['n_epochs']
    niter = len(trainloader)
    STATUS = {'val_acc':-1., 'best_val_acc':-1.}
    for e in range(epoch):
        net.train()
        for i, (x,y0) in enumerate(trainloader):
            net.zero_grad()

            x = x.to(torch.float).to(device=device)
            y0 = y0.to(device=device)
            # print(x.shape, y0) # torch.Size([16, 16]) tensor([3, 5, 4, 2, 6, 6, 4, 4, 5, 6, 6, 0, 5, 0, 0, 1], device='cuda:0')

            y = net(x)

            loss = criterion(y,y0)
            loss.backward()
            optimizer.step()            

            if (i+1)==niter or (i+1)%10==0:
                update_text = f'epoch {e}: {i+1}/{niter} | val acc:{round(STATUS["val_acc"],3)}'
                print('%-64s'%(update_text), end='\r')

                if DEBUG_TOGGLES['trainval_iter']: break

        VAL_STATUS = drybean_validate(net, valloader)
        STATUS["val_acc"] = VAL_STATUS['val_acc']
        if VAL_STATUS['val_acc'] > STATUS['best_val_acc']:
            torch.save({'model_state_dict': net.state_dict()}, DIRS['MODEL_DIR'])            
            STATUS.update({
                '_args': dargs,
                'best_val_acc': VAL_STATUS['val_acc'], 
                'n_params': n_params, 
                })
            with open(DIRS['TRAINVAL_STATUS_DIR'], 'w') as json_file:
                json.dump(STATUS, json_file, indent=4, sort_keys=True)

            if VAL_STATUS['val_acc']> dargs['VAL_TARGET']: 
                print(f"\nVALIDATION TARGET {dargs['VAL_TARGET']} reached at epoch {e}.")
                break

    end = time.time()
    elapsed = end - start
    print(' time taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), 
        str(round(elapsed/60.,1)), 
        str(round(elapsed/3600.,1))))

    print('\ntraining ended')



def drybean_validate(net, valloader):
    net.eval()

    n_correct, n_total = 0,0
    n_positive = 0
    for i, (x,y0) in enumerate(valloader):
        x = x.to(torch.float).to(device=device)
        y0 = y0.to(device=device)
        y = net(x)

        y_pred = torch.argmax(y,dim=1).detach().cpu().numpy()
        y0 = y0.clone().detach().cpu().numpy()
        
        # print(y_pred, y0)
        # [5 6 5 4 4 2 4 4 0 5 0 0 4 0 4 5] [6 6 5 4 4 2 4 4 0 1 0 0 6 0 4 1]
        # print(y_pred==y0)
        # [False  True  True  True  True  True  True  True  True False  True  True

        n_correct += sum(y_pred==y0)
        n_total += len(y0)

        if DEBUG_TOGGLES['trainval_iter']:
            if i>10: break

    val_acc = n_correct/n_total
    
    VAL_STATUS = {
        'val_acc': val_acc,
    }
    return VAL_STATUS