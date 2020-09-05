from utils.utils import *

"""
pm is printing manager from utils/printing_manager.py
"""

class StateTracker(object):
	"""
	assume state tracker is saved in the following format:
		ST.series_name.X.state <-- (self.ObjectMarkerName, series_name, n_th_run, file_extension)

	the n_th refers to the n_th time the training script is run.
		Recording by the n_th run is useful for logging purpose.
	"""
	def __init__(self, **kwargs):
		super(StateTracker, self).__init__()

		# Init args
		mode = kwargs.get('mode', 'load_latest_if_available')
		self.ckpt_path = kwargs.get('ckpt_path')
		self.training_series_name = kwargs.get('training_series_name')
		self.load_this_n_th_run = kwargs.get('load_this_n_th_run') 
		self.verbose = kwargs.get('verbose')
		self.tab_level = kwargs.get('tab_level')

		# Properties
		self.ObjectMarkerName = 'ST'

		# Init
		pm.printvm('StateTracker(). Initializing'%(),
			tab_level=self.tab_level,verbose=self.verbose, verbose_threshold=50)
		self.manage_directories()
		self.current_run = None 	# if no saved run, this is 0 	(count starting with 1)

		if mode=='load_latest_if_available':
			self.latest_saved_run = self.find_latest_save_file()
		elif mode=='load_at_n_th_run':
			pm.printvm('StateTracker(). mode: load_at_n_th_run. Use load_instance() externally!**'%(),
				tab_level=self.tab_level+1,verbose=self.verbose, verbose_threshold=50)
			self.current_run = self.load_this_n_th_run
		else:
			pm.printvm('StateTracker(). No load mode selected.'%(),
				tab_level=self.tab_level+1,verbose=self.verbose, verbose_threshold=50)
			

	def load_latest_if_available(self, latest_saved_run, for_training=True):
		pm.printvm('StateTracker(). load_latest_if_available(). latest_saved_run:%s'%(str(latest_saved_run)),
			tab_level=self.tab_level,verbose=self.verbose, verbose_threshold=200)
		if latest_saved_run is None:
			self.init_new_instance()
			return self
		else:
			st = self.load_instance(latest_saved_run, for_training=for_training)
			st.latest_saved_run = latest_saved_run
			return st

	def display_simple_state(self, tab_level=0, verbose=0):
		pm.printvm('StateTracker(). display_simple_state()'%(),
			tab_level=self.tab_level,verbose=self.verbose, verbose_threshold=50)
		pm.printvm('current_run:%s\nckpt_path:%s\ntraining_series_name:%s'%(
			str(self.current_run),
			str(self.ckpt_path),
			str(self.training_series_name)
			), tab_level=self.tab_level+1, verbose=self.verbose, verbose_threshold=50)

	def display_end_state(self, tab_level=0, verbose=0):
		pm.printvm('StateTracker(). display_end_state()'%(),
			tab_level=tab_level,verbose=self.verbose, verbose_threshold=50)
		list_of_all_runs = [xkey for xkey in self.save_data_by_nth_run]
		list_of_all_epochs = [xkey for xkey in self.save_data_by_epoch]

		pm.print_2Dmatrix_format(list_of_all_runs, header='runs:', cell_string_format='%s', column_size=10, separator=',',
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		pm.print_2Dmatrix_format(list_of_all_epochs, header='epochs:', cell_string_format='%s', column_size=10, separator=',',
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)

	def manage_directories(self):
		assert(self.ckpt_path is not None)
		assert(self.training_series_name is not None)
		if not os.path.exists(self.ckpt_path):
			os.mkdir(self.ckpt_path)
		self.series_folder_path = os.path.join(self.ckpt_path, self.training_series_name)
		if not os.path.exists(self.series_folder_path):
			os.mkdir(self.series_folder_path)

	def find_latest_save_file(self):
		latest_saved_run= None 
		pm.printvm('StateTracker().find_latest_save_file(). List of state files:'%(),
			tab_level=self.tab_level,verbose=self.verbose, verbose_threshold=50)
		all_state_files = os.listdir(self.series_folder_path)
		is_empty = True
		for x in all_state_files:
			temp = x.split('.')
			if len(temp) == 4 and temp[-1]=='state':
				pm.printvm('%s'%(str(x)),
					tab_level=self.tab_level+1,verbose=self.verbose, verbose_threshold=50)
				prefix, training_series_name, this_saved_run, file_extension = temp
				
				if latest_saved_run is None:
					latest_saved_run = this_saved_run
				else:
					WATERMARK = 'InOrder' # just for checking
					if int(this_saved_run)>int(latest_saved_run):
						latest_saved_run = this_saved_run
					else:
						WATERMARK = 'NotOrdered'
					pm.printvm('SIMPLEWaterMark:%s'%(WATERMARK),
						tab_level=self.tab_level+2,verbose=self.verbose, verbose_threshold=50)
				assert(training_series_name == self.training_series_name)
				assert(prefix == self.ObjectMarkerName)
				is_empty = False
		if is_empty: 
			pm.printvm('(empty)'%(), tab_level=self.tab_level+1,verbose=self.verbose, verbose_threshold=50)

		return latest_saved_run

	def init_new_instance(self):
		pm.printvm('init_new_instance(). '%(),
			tab_level=self.tab_level+1,verbose=self.verbose, verbose_threshold=50)
		self.save_data_by_nth_run = {1:{}}
		self.save_data_by_epoch = {}
		self.current_run = 1	
		self.setup_for_this_run(self.current_run, -1, 0,None)

	def load_instance(self, n_th_run, for_training=True):
		# e.g. ST.training_series_name.5.state
		state_name = self.ObjectMarkerName + '.' + str(self.training_series_name) + '.' + str(n_th_run) + '.state'
		full_save_dir = os.path.join(self.series_folder_path, state_name)

		pm.printvm('load_instance(). state_name:%s '%(str(state_name)),
			tab_level=self.tab_level+1,verbose=self.verbose, verbose_threshold=50)
		pm.printvm('  path: %s '%(str(self.series_folder_path)),
			tab_level=self.tab_level+1,verbose=self.verbose, verbose_threshold=50)
		pkl_file = open(full_save_dir, 'rb')
		st = pickle.load(pkl_file)
		pkl_file.close()

		if for_training:
			prev_run = st.current_run
			latest_saved_epoch = st.save_data_by_nth_run[prev_run]['latest_saved_epoch']
			total_iteration = st.save_data_by_nth_run[prev_run]['total_iteration']
			st.setup_for_this_run(prev_run+1, latest_saved_epoch, total_iteration, None)
		st.current_run += 1
		return st

	#################### Operational functions ####################

	def setup_for_this_run(self, n_th_run, latest_saved_epoch, total_iteration,config_data):
		self.save_data_by_nth_run[n_th_run] = {
			'latest_saved_epoch': latest_saved_epoch, # zero based
			'total_iteration': total_iteration, 
			'config_data': config_data
		}

	def setup_for_this_epoch(self, n_epoch, tab_level=None, verbose=None):
		if tab_level is None: 
			tab_level = self.tab_level
		if verbose is None:
			verbose = self.verbose
		pm.printvm('setup_for_this_epoch(). n_epoch:%s [zero-based] or %s [one-based]'%(str(n_epoch),str(n_epoch+1)),
			tab_level=tab_level,verbose=verbose, verbose_threshold=50)
		self.save_data_by_epoch[n_epoch] = {
			'loss': []
		}

	def get_latest_saved_epoch(self):
		return self.save_data_by_nth_run[self.current_run]['latest_saved_epoch']

	def update_epoch(self):
		self.save_data_by_nth_run[self.current_run]['latest_saved_epoch'] +=1

	def store_loss_by_epoch(self, loss, n_epoch):
		self.save_data_by_epoch[n_epoch]['loss'].append(loss)

	def update_state(self, total_iter_in_this_run, config_data):
		self.save_data_by_nth_run[self.current_run]['total_iteration'] += total_iter_in_this_run
		self.save_data_by_nth_run[self.current_run]['config_data'] = config_data

		# e.g. ST.training_series_name.5.state
		state_name = self.ObjectMarkerName + '.' + str(self.training_series_name) + '.' + str(self.current_run) + '.state'
		full_save_dir = os.path.join(self.series_folder_path, state_name)

		pm.printvm('StateTracker().update_state()\n  full_save_dir:%s'%(str(full_save_dir)),
			tab_level=self.tab_level,verbose=self.verbose, verbose_threshold=50)

		output = open(full_save_dir, 'wb')
		pickle.dump(self, output)
		output.close()
