# most of the code here is directly adapted from
# https://github.com/ericotjo001/explainable_ai/tree/master/gax/src
# Also, the relevant dataset can be downloaded from
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

import os
import numpy as np
import pandas as pd

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
    
    # MODEL_DIR = os.path.join(PROJECT_DIR, 'sqann.model')
    
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

        # 'MODEL_DIR': MODEL_DIR, 
    }
    return DIRS


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