from utils.utils import *

class ResultWrapper(object):
	def __init__(self, **kwargs):
		super(ResultWrapper, self).__init__()
		
		# The item will be saved in
		# working_dir/path_to_folder/filename
		self.working_dir = None
		self.path_to_folder = None 
		self.filename = None
		self.fullpath = kwargs.get('fullpath', None)

		# self.result # define your own result format

	def save_result(self, working_dir, path_to_folder, filename, 
		verbose=0, tab_level=0,verbose_threshold=50):
		self.fullpath = os.path.join(working_dir, path_to_folder, filename)
		pm.printvm('ResultWrapper(). saveresult()\n  path:%s'%(str(self.fullpath)), 
			tab_level=tab_level,verbose=verbose, verbose_threshold=verbose_threshold)
		
		output = open(self.fullpath, 'wb')
		pickle.dump(self, output)
		output.close()

	def load_result(self, working_dir, path_to_folder, filename, 
		verbose=0, tab_level=0,verbose_threshold=50):
		self.fullpath = os.path.join(working_dir, path_to_folder, filename)
		pm.printvm('ResultWrapper(). load_result()\n  path:%s'%(str(self.fullpath)), 
			tab_level=tab_level,verbose=verbose, verbose_threshold=verbose_threshold)

		pkl_file = open(self.fullpath, 'rb')
		res = pickle.load(pkl_file)
		pkl_file.close()
		return res