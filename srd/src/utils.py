import os, pickle
from .printing_manager import ShortPrint
import numpy as np
sp = ShortPrint()



class FastPickleClient(object):
    def __init__(self):
        super(FastPickleClient, self).__init__()
        self.save_text = 'Saving data via FastPickleClient...'
        self.load_text = 'Loading data via FastPickleClient...'
    
    def pickle_data(self, save_data, save_dir, tv=(0,0,None), text=None):
        if text is not None: 
            self.save_text = text
        output = open(save_dir, 'wb')
        pickle.dump(save_data, output)
        output.close()      
        sp.prints('%s\n  %s'%(str(self.save_text),str(save_dir)), tv=tv)

    def load_pickled_data(self, pickled_dir, tv=(0,0,None), text=None):
        if text is not None:
            self.load_text = text
        pkl_file = open(pickled_dir, 'rb')
        this_data = pickle.load(pkl_file)
        pkl_file.close()        
        sp.prints('%s\n  %s'%(str(self.load_text),str(pickled_dir)), tv=tv)
        return this_data


def average_every_n(x, n=10):
    if not len(x)%n==0:
        x = list(x) + [x[-1] for i in range(n-len(x)%n)]
    x = np.array(x).reshape(-1,n)
    return np.mean(x, axis=1)
