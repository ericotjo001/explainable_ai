import os, pickle
from utils.printing_manager import ShortPrint
sp = ShortPrint() 

def check_default_aux_folders():
	DEFAULT_AUX_FOLDER = ['checkpoint', 'checkpoint/cache']
	for folder_name in DEFAULT_AUX_FOLDER:
		if not os.path.exists(folder_name):
			os.mkdir(folder_name)
		else:
			pass # print(folder_name,'OK')

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

