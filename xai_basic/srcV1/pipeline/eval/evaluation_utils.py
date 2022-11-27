import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils import FastPickleClient
import matplotlib.pyplot as plt

class EvalMidTraining(FastPickleClient):
    def __init__(self,):
        super(EvalMidTraining, self).__init__()
        self.iter = 0
        self.model_last_saved_iter = 0

        # FastPickleClient
        self.save_text = 'Saving evaluator via EvalMidTraining(FastPickleClient)...'
        self.load_text = 'Loading evaluator via EvalMidTraining(FastPickleClient)...'
   
    # inherited from FastPickleClient
    # def pickle_data(self, save_data, save_dir, tv=(0,0,None)):
    # def load_pickled_data(self, pickled_dir, tv=(0,0,None)):

    def setup(self, avg_loss_every_n_iter=24):
        self.avg_loss_every_n_iter = avg_loss_every_n_iter 
        self.running_loss = 0.
        self.loss_iter, self.losses = [],[] # x,y -axes to plot loss
                 
    def compute_running_average_loss(self, loss):
        self.running_loss += (float(loss)/self.avg_loss_every_n_iter)
        if self.iter%self.avg_loss_every_n_iter==0:
            self.loss_iter.append(self.iter)
            self.losses.append(self.running_loss)
            self.running_loss = 0.

def pytorch_singleton_to_batch_reshape(x,this_device=None):
    return torch.tensor(x.reshape((1,)+x.shape)).to(torch.float).to(this_device)


def eval_plot_loss(initiate_or_load_model, config_data, **kwargs):
    from pipeline.training.training_utils import prepare_save_dirs 
    MODEL_DIR, INFO_DIR, CACHE_FOLDER_DIR = prepare_save_dirs(config_data)
    net, evaluator = initiate_or_load_model(MODEL_DIR, INFO_DIR, config_data)
    
    plot_mode = kwargs['plot_mode'] if 'plot_mode' in kwargs else None
    plt.figure()
    ax = plt.gcf().add_subplot(111) 
    ax.plot(evaluator.loss_iter, evaluator.losses)
    ax.set_xlabel('iter')
    ax.set_ylabel('loss')

    if plot_mode is None:
        plt.show()
    elif plot_mode == 'savefig':
        FIGURE_DIR = os.path.join('checkpoint', config_data['model_name'], config_data['model_name'] + '.jpg')
        plt.savefig(FIGURE_DIR, )