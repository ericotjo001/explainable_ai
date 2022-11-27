import os, joblib, random, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

os.environ['KMP_DUPLICATE_LIB_OK']='True' # prevent weird double import error

def makedirifnot(THIS_DIR):
    if not os.path.exists(THIS_DIR):
        os.makedirs(THIS_DIR, exist_ok=True)


def manage_dir(dargs):
    CKPT_FOLDER_DIR = dargs['CKPT_FOLDER_DIR']
    makedirifnot(CKPT_FOLDER_DIR)
    PROJECT_FOLDER_DIR = os.path.join(CKPT_FOLDER_DIR, dargs['PROJECT_NAME'])
    makedirifnot(PROJECT_FOLDER_DIR)

    model_name = dargs['model_name'] if 'model_name' in dargs else 'dummy'

    MODEL_DIR = os.path.join(PROJECT_FOLDER_DIR, f'{model_name}.pt')
    MODEL_INFO_DIR = os.path.join(PROJECT_FOLDER_DIR, f'{model_name}_info.json')
    LOSS_INFO_DIR = os.path.join(PROJECT_FOLDER_DIR, f'{model_name}_loss_info.json')
    HEATMAP_SAMPLE_DIR = os.path.join(PROJECT_FOLDER_DIR, f'{model_name}_heatmaps.png')


    SHARD_FOLDER_DIR = os.path.join(PROJECT_FOLDER_DIR, 'data.shards')
    SHARD_VAL_FOLDER_DIR = os.path.join(PROJECT_FOLDER_DIR, 'dataval.shards')
    SHARD_TEST_FOLDER_DIR = os.path.join(PROJECT_FOLDER_DIR, 'datatest.shards')
    makedirifnot(SHARD_FOLDER_DIR)
    makedirifnot(SHARD_VAL_FOLDER_DIR)
    makedirifnot(SHARD_TEST_FOLDER_DIR)

    METRIC_AGGREGATE_DIR = os.path.join(PROJECT_FOLDER_DIR, f'{model_name}_metric_aggreagate.csv')

    DIRS = {
        'CKPT_FOLDER_DIR': CKPT_FOLDER_DIR,
        'PROJECT_FOLDER_DIR': PROJECT_FOLDER_DIR,

        'MODEL_DIR': MODEL_DIR,
        'MODEL_INFO_DIR': MODEL_INFO_DIR,
        'LOSS_INFO_DIR':LOSS_INFO_DIR,
        'HEATMAP_SAMPLE_DIR': HEATMAP_SAMPLE_DIR, 

        'SHARD_FOLDER_DIR': SHARD_FOLDER_DIR,
        'SHARD_VAL_FOLDER_DIR': SHARD_VAL_FOLDER_DIR,
        'SHARD_TEST_FOLDER_DIR': SHARD_TEST_FOLDER_DIR,

        'METRIC_AGGREGATE_DIR': METRIC_AGGREGATE_DIR
    }
    return DIRS 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_batch_one_tensor(x, device=None):
    # x is a numpy array in (C,H,W)
    return torch.tensor(x).to(torch.float).unsqueeze(0).to(device=device)


