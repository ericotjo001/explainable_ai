# most of the code here is directly adapted from
# https://github.com/ericotjo001/explainable_ai/tree/master/gax/src
# Also, the relevant dataset can be downloaded from
# https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset

import os
import numpy as np
import pandas as pd

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
    }
    return DIRS


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