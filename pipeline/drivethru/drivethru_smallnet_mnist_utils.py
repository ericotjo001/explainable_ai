from utils.utils import *
from pipeline.data.load_data import load_mnist_0001, AutoReloaderTestMNIST
from utils.loss import compute_loss
from utils.state_tracker import StateTracker
import pipeline.training.train_smallnet_mnist as trsm



###############################################################
# loader utils
###############################################################

def new_or_load_model(state_tracker, config_data, verbose=0, tab_level=0):
	from models.networks import SmallCNNprobe0001
	IMG_SIZE=(28, 28)
	if config_data['data_from_torch']['mnist']['resize'] is None:
		net = SmallCNNprobe0001(IMG_SIZE=IMG_SIZE,relprop_mode=config_data['lrp']['relprop_mode'], verbose=verbose, tab_level=tab_level)
	else:
		IMG_SIZE = config_data['data_from_torch']['mnist']['resize']
		net = SmallCNNprobe0001(IMG_SIZE=IMG_SIZE, relprop_mode=config_data['lrp']['relprop_mode'], 
			set_default_params=False, verbose=verbose, tab_level=tab_level)
		if IMG_SIZE == (140,140):
			from models.networks_smallCNN import SmallCNN_custom_setting0001
			net = SmallCNN_custom_setting0001(net, IMG_SIZE)
		else:
			raise Exception('Please manually set params')
				
	net = new_or_load_model_inner(state_tracker, net, verbose=verbose, tab_level=tab_level)
	return net



def setup_state_tracker(config_data, tracker_name, for_training=True, verbose=0, tab_level=0):
	ckpt_path = os.path.join(config_data['working_dir'],'checkpoint')
	training_series_name = tracker_name + str(config_data['training']['series_name'])
	state_tracker = DrivethruTracker(
		mode='load_latest_if_available',
		ckpt_path=ckpt_path,
		training_series_name=training_series_name,
		load_this_n_th_run=None,
		verbose=verbose,
		tab_level=tab_level+1)

	state_tracker = state_tracker.load_latest_if_available(state_tracker.latest_saved_run,for_training=for_training)

	##########################################################	
	# update after loading> This is for when directory has changed
	state_tracker.ckpt_path = os.path.join(config_data['working_dir'],'checkpoint')
	state_tracker.series_folder_path = os.path.join(state_tracker.ckpt_path, state_tracker.training_series_name)
	##########################################################

	state_tracker.display_simple_state(tab_level=tab_level+1, verbose=verbose)	
	return state_tracker	

def setup_training_and_data_loader(config_data, verbose=0, tab_level=0):
	config_data['data_from_torch']['mnist']['training_mode'] = True
	data_loader = load_mnist_0001(config_data, verbose=0)
	n_batch_train = len(data_loader)
	pm.printvm("setup_training_and_data_loader()"%(),
		tab_level=tab_level,verbose=verbose, verbose_threshold=0)
	pm.printvm("n_batch_train = len(data_loader):%s"%(str(n_batch_train)),
		tab_level=tab_level+1,verbose=verbose, verbose_threshold=0)
	return data_loader, n_batch_train

def new_or_load_model_inner(state_tracker, net, verbose=0, tab_level=0):
	pm.printvm("new_or_load_model(). latest_saved_run:%s"%(str(state_tracker.latest_saved_run)),
		 tab_level=tab_level,verbose=verbose, verbose_threshold=50)
	if state_tracker.latest_saved_run is None:
		pm.printvm("Init new model!"%(),
			tab_level=tab_level+1,verbose=verbose, verbose_threshold=50)
	else:
		pm.printvm("Loading model... prev_run:%s"%(str(state_tracker.current_run-1)),
			tab_level=tab_level+1,verbose=verbose, verbose_threshold=50)
		net.load_state_dict(torch.load(trsm.get_path_to_load_model(state_tracker, 
			verbose=verbose, tab_level=tab_level+1)))
	return net

class DrivethruTracker(StateTracker):
	def __init__(self, **kwargs):
		pm.printvm('DrivethruTracker(). Initializing'%(),
			tab_level=kwargs['tab_level'],verbose=kwargs['verbose'], verbose_threshold=50)
		self.save_data_by_iter_details = {
			# n_epoch :{
			#	 total_iter_in_this_run	:{
			#		'any data': 'some data'
			# 	}
			# }
		}
		kwargs['tab_level'] += 1
		super(DrivethruTracker, self).__init__(**kwargs)


###############################################################
# debug utils
###############################################################

def set_debug_config(IS_DEBUG, config_data):
	if IS_DEBUG:
		config_data['general']['epoch'] = 2
		config_data['drivethru']['no_of_evals_per_run'] = 2
	return config_data

def print_progress_percentage(i, progress_tracker, n_batch_train, verbose=250, tab_level=0):
	if (i+1)%progress_tracker==0: 
		pm.printvm('\n%s percent\n'%(str(round(100*i/n_batch_train,1))),
			verbose=verbose, tab_level=tab_level, verbose_threshold=20)

def get_eval_every_n_iter(config_data, n_batch_train, n_epoch,
	NO_OF_EVALUATION_DESIRED=2, 
	manual_specification=True,
	DEBUG_N_ITER_MAX_PER_EPOCH=0):
	if manual_specification:
		eval_every_n_iter = config_data['drivethru']['eval_every_n_iter']
	else:
		if DEBUG_N_ITER_MAX_PER_EPOCH==0:
			# not debug
			total_iter = n_epoch*n_batch_train
			eval_every_n_iter = int(np.floor(total_iter/NO_OF_EVALUATION_DESIRED))
		else:
			total_iter_debug = n_epoch*DEBUG_N_ITER_MAX_PER_EPOCH
			eval_every_n_iter = int(np.floor(total_iter_debug/NO_OF_EVALUATION_DESIRED))			
	return eval_every_n_iter


def DEBUG_train_loop_0002(DEBUG_N_ITER_MAX_PER_EPOCH, n_iter, n_epoch ,tab_level=0, verbose=0):
	DEBUG_SIGNAL = [0,0]
	# n_epoch n_iter are both zero based
	if DEBUG_N_ITER_MAX_PER_EPOCH>0:
		N_EPOCH_MAX = 9999
		PRINT_EVERY_N_EPOCH = 1
		if n_iter >= DEBUG_N_ITER_MAX_PER_EPOCH-1:
			DEBUG_SIGNAL[0] = 1
			if (n_epoch+1)%PRINT_EVERY_N_EPOCH==0:
				pm.printvm('drivethru_0001_smallnet_mnist.py. DEBUG_train_loop_0002(). MAX_iter_REACHED. n_epoch:%s n_iter:%s'%(
					str(n_epoch), str(n_iter+1)), tab_level=tab_level+1, tab_shape='  ',verbose=0, verbose_threshold=None)		
		if n_epoch > N_EPOCH_MAX:
			DEBUG_SIGNAL[1] = 1
			if n_iter == 0:
				pm.printvm('drivethru_0001_smallnet_mnist.py. DEBUG_train_loop_0002(). Now at MAX_EPOCH. n_epoch:%s'%(str(n_epoch)),
					tab_level=tab_level, tab_shape='  ',verbose=0, verbose_threshold=None)
	return DEBUG_SIGNAL

def DEBUG_COMPUTE_NUMBER_OF_DRIVETHRU(DEBUG_COMPUTE_NUMBER_OF_DRIVETHRU_SIGNAL,
	eval_every_n_iter, n_batch_train, n_epoch):
	DEBUG_SIGNAL = 0
	if DEBUG_COMPUTE_NUMBER_OF_DRIVETHRU_SIGNAL:
		print('DEBUG_COMPUTE_NUMBER_OF_DRIVETHRU()')
		print('  eval_every_n_iter:%s'%(str(eval_every_n_iter)))
		print('  n_epoch:%s\n  batch_train:%s / %s'%(str(n_epoch),
			str(n_batch_train),str(DEBUG_N_ITER_MAX_PER_EPOCH)))
		total_iter = n_epoch*n_batch_train
		total_iter_debug = n_epoch*DEBUG_N_ITER_MAX_PER_EPOCH
		print('  total iter:%s / %s'%(str(total_iter),str(total_iter_debug)))
		print('  eval_every_n_iter:%s'%(str(eval_every_n_iter)))
		print('  total no of eval:%s / %s'%(str(np.floor(total_iter/eval_every_n_iter)),
			str(np.floor(total_iter_debug/eval_every_n_iter))))
		DEBUG_SIGNAL = True
	return DEBUG_SIGNAL