from utils.utils import *
from utils.result_wrapper import ResultWrapper


def visual_final_LRP_output(config_data, tab_level=0, verbose=0):
	print('visual_final_LRP_output(). HARDCODED VARIABLES? YES.')
	SHOW_CASE_INDEX = slice(0,10,None)
	N_th_RUN_TO_LOAD = config_data['visual']['n_th_run_to_load']

	from pipeline.lrp.lrp_smallnet_mnist import prepare_state_tracker_for_use 
	state_tracker = prepare_state_tracker_for_use(N_th_RUN_TO_LOAD, config_data, verbose=verbose, tab_level=tab_level)
	
	visual_mnist_lrp_abstract_model(config_data, SHOW_CASE_INDEX, N_th_RUN_TO_LOAD,
		state_tracker, tab_level=0, verbose=250)


#########################################################################
# Abstracted functions to visualize mnist lrp i.e. not model-specific
#########################################################################

def visual_mnist_lrp_abstract_model(config_data, SHOW_CASE_INDEX, N_th_RUN_TO_LOAD,
	state_tracker, tab_level=0, verbose=0):
	RELPROP_MODE = config_data['lrp']['relprop_mode']

	res = ResultWrapper()
	working_dir = config_data['working_dir']
	path_to_folder = os.path.join('checkpoint',state_tracker.training_series_name)
	filename = str(RELPROP_MODE) + '_' + str(N_th_RUN_TO_LOAD) + '.result'
	res = res.load_result(working_dir, path_to_folder, filename, 
		verbose=250, tab_level=0,verbose_threshold=50)

	n = len(res.lrp_output)
	pm.printvm('%10s| %-15s | %-7s | %-7s | %-15s | %-10s, %-10s'%(
		'','x.shape','y','y0','R.shape', 'R max', 'R min'),
		tab_level=tab_level,verbose=verbose, verbose_threshold=None)

	display = res.lrp_output[SHOW_CASE_INDEX]
	x_stack, R_stack, R1_stack, y_stack, gt_stack = \
		create_stack(display,n, R1_clamp_factor=1.,tab_level=tab_level,verbose=verbose)
	plot_stack(x_stack, R_stack, R1_stack, y_stack, gt_stack)

	R1_clamp_factor_collection = [1.0, 0.5, 0.1]
	plot_and_compare_normalize_stacks(display, n, R1_clamp_factor_collection,
		tab_level=tab_level, verbose=verbose)
	plt.show()

def create_stack(display,n, R1_clamp_factor=1.0, tab_level=0,verbose=0):
	img_stack, gt_stack, y_stack = [], [], []
	for i, xyR in enumerate(display):
		x,y,y0,R = xyR
		y1 = np.argmax(y)
		pm.print_in_loop('%-10s| %15s | %7s | %7s | %15s | %10s, %10s'%(str(i),
			str(x.shape),str(y1),str(y0[0]),str(R.shape),str(np.max(R)),str(np.min(R))
			), i, n, first=3, last=1,
			tab_level=tab_level,verbose=verbose, verbose_threshold=None)
		R1 = normalize_numpy_array(R,target_min=-1,target_max=1, 
			source_min=None, source_max=None, verbose = 0)
		R1 = np.clip(R1, -R1_clamp_factor, R1_clamp_factor)
		img_stack.append([x,R,R1])
		y_stack.append(y1)
		gt_stack.append(y0[0])

	assert(len(img_stack)>0)
	x_stack = np.concatenate([xR[0] for xR in img_stack],3)
	R_stack = np.concatenate([xR[1] for xR in img_stack],3)
	R1_stack = np.concatenate([xR[2] for xR in img_stack],3)
	# print('img_stack.shape:%s'%(str(img_stack.shape)))
	return x_stack, R_stack, R1_stack, y_stack, gt_stack

def plot_stack(x_stack, R_stack, R1_stack, y_stack, gt_stack):
	fig = plt.figure()
	ax = fig.add_subplot(311)
	im = ax.imshow(x_stack.squeeze())
	plt.title('y :%s\ngt:%s'%(str(y_stack),str(gt_stack)))
	plt.colorbar(im)
	plt.axis('off')

	ax1 = fig.add_subplot(312)
	im1 = ax1.imshow(R_stack.squeeze(),cmap='bwr')
	plt.colorbar(im1)
	plt.title('R')
	plt.axis('off')

def plot_and_compare_normalize_stacks(display, n, R1_clamp_factor_collection, tab_level=0,verbose=0):	
	nc = len(R1_clamp_factor_collection)
	fig = plt.figure()
	
	for i, Rc in enumerate(R1_clamp_factor_collection):
		pos = (nc, 1, i+1)
		x_stack, R_stack, R1_stack, y_stack, gt_stack = \
			create_stack(display,n, R1_clamp_factor=Rc,tab_level=tab_level,verbose=verbose)
		plot_stack2(R1_stack, Rc, pos, fig)
		plt.title('normalized R. Factor:%s'%(str(Rc)))
	

def plot_stack2(R1_stack, R1_clamp_factor, pos, fig):
	ax2 = fig.add_subplot(pos[0],pos[1],pos[2])
	im2 = ax2.imshow(R1_stack.squeeze(),cmap='bwr')
	plt.colorbar(im2)
	plt.axis('off')	