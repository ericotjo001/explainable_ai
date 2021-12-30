import os
import numpy as np
import matplotlib.pyplot as plt

from .robot import ResultsData
class ResultsDataLava(ResultsData):
    def __init__(self, ):
        super(ResultsDataLava, self).__init__()
        self.tile_names = ['grass','dirt','unrecognized']
        self.results = {tile_name:[] for tile_name in self.tile_names}
        self.n_correct = 0
        self.n_total = 0

    # def pickle_data(self, save_data, save_dir, tv=(0,0,None), text=None):
    # def load_pickled_data(self, pickled_dir, tv=(0,0,None), text=None):

    # def add_result(self, tiles, reached): # this function is inherited

    def save_histogram(self, IMG_DIR, iter_limit=48, n_max=None):
        plt.figure(figsize=(8,4))

        ax = plt.gcf().add_subplot(131)
        vn,vx,_ = plt.gca().hist( self.results['grass'], bins=range(iter_limit+1), rwidth=0.5 )
        plt.gca().set_xlabel('grass. max:%s'%(str(np.max(vn))))
        this_max = [vn]

        ax2 = plt.gcf().add_subplot(132)
        vn,vx,_ = plt.gca().hist( self.results['dirt'], bins=range(iter_limit+1), rwidth=0.5 )
        this_max.append(vn)
        plt.gca().set_xlabel('dirt. max:%s'%(str(np.max(vn))))

        ax3 = plt.gcf().add_subplot(133)
        vn,vx,_ = plt.gca().hist( self.results['unrecognized'], bins=range(iter_limit+1), rwidth=0.5 )
        this_max.append(vn)
        plt.gca().set_xlabel('unrecognized. max:%s'%(str(np.max(vn))))

        n_max = int(np.max(this_max) *1.1)
        ax.set_ylim((0,n_max))
        ax2.set_ylim((0,n_max))

        plt.savefig(IMG_DIR)
        plt.close()

from .robot import run_robot_eval, run_robot_srd_training
def run_robot_eval_lava(args, run_srd=False,):
    run_robot_eval(args, run_srd=run_srd, ResultsData=ResultsDataLava)

def run_robot_srd_training_lava(args):
    run_robot_srd_training(args, ResultsData=ResultsDataLava)