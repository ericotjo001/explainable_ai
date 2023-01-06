import os, pickle

def create_if_not_exists(this_dir):
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)

from .printing_manager import ShortPrint
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

fastpickle = FastPickleClient()