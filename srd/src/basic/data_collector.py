import matplotlib.pyplot as plt
from src.utils import average_every_n
from ..utils import FastPickleClient

class FishUnitDataCollector(FastPickleClient):
    def __init__(self, args):
        super(FishUnitDataCollector, self).__init__()
        self.iter = []
        self.energy = []
        self.tile = []

    # def pickle_data(self, save_data, save_dir, tv=(0,0,None), text=None):
    # def load_pickled_data(self, pickled_dir, tv=(0,0,None), text=None):

    def get_unit_data(self, i, fish, ENV):
        self.iter.append(i)
        self.energy.append(fish.INTERNAL_STATE['energy'])
        self.tile.append(ENV[0])

    def display_data(self, start_index=0, end_index=None, save_dir=None, average_every=0):
        plt.figure(figsize=(5,4))
        plt.gcf().add_subplot(111)
        if average_every==0:
            plt.gca().plot(self.iter[start_index:end_index],self.energy[start_index:end_index], label='energy')
            plt.gca().plot(self.iter[start_index:end_index],self.tile[start_index:end_index], c='r', label='current tile',linewidth=0.3)
            plt.gca().set_ylim([0,1])
            plt.gca().set_xlabel('iter')
        else:
            iters = average_every_n(self.iter, n=average_every).astype(int)
            plt.gca().plot(iters, average_every_n(self.energy, n=average_every), label='energy')
            plt.gca().set_ylim([0,1])
        plt.legend()
        if save_dir is None:
            plt.show()
        else:
            plt.savefig(save_dir)


    def display_srd_data(self, args, save_dir=None):
    
        plt.figure(figsize=(12,4))
        plt.gcf().add_subplot(131)
        plt.gca().plot(self.iter[:128],self.energy[:128], label='energy')
        plt.gca().plot(self.iter[:128],self.tile[:128], c='r', label='current tile',linewidth=0.3)
        plt.gca().set_ylim([0,1])

        plt.gcf().add_subplot(132)
        plt.gca().plot(self.iter[-128:],self.energy[-128:], label='energy')
        plt.gca().plot(self.iter[-128:],self.tile[-128:], c='r', label='current tile',linewidth=0.3,)
        plt.gca().set_ylim([0,1])
        plt.gca().set_yticks([])
        plt.gca().set_xlabel('iter')
        plt.legend()

        plt.gcf().add_subplot(133)
        iters = average_every_n(self.iter, n=args['average_every']).astype(int)
        plt.gca().plot(iters, average_every_n(self.energy, n=args['average_every']), label='avg energy')
        plt.gca().set_ylim([0,1])

        plt.legend()

        if save_dir is None:
            plt.show()
        else:
            plt.savefig(save_dir)