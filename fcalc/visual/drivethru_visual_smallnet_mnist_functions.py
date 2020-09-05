from utils.utils import *

#################################################################
# Display functions
#################################################################
def visualize_mean_power_by_layer_over_iter(processed_datapoint_vs_iter, path_to_save_folder, 
	LIST_OF_LAYERS_TO_OBSERVE, LIST_OF_DATA_NAME, verbose=0, tab_level=0):
	pm.printvm('visualize_mean_power_by_layer_over_iter().', verbose=verbose, tab_level=tab_level, verbose_threshold=100)
	print_processed_datapoint_vs_iter(processed_datapoint_vs_iter,verbose=50, tab_level=tab_level+1)

	from visual.visual_data_extract import VisualDataExtract
	vde = VisualDataExtract()
	for LAYER_NAME in LIST_OF_LAYERS_TO_OBSERVE:
		for data_name in LIST_OF_DATA_NAME:
			mean_power_vs_iter_this_layer = vde.extract_data_by_iter_by_sample_list_0001(
				processed_datapoint_vs_iter, LAYER_NAME, data_name,
				verbose=verbose, tab_level=tab_level+1)
			plot_mean_power_vs_iter(mean_power_vs_iter_this_layer, path_to_save_folder, 
				filename_appendix=data_name+'_'+str(LAYER_NAME), verbose=verbose, tab_level=tab_level+2)


def plot_mean_power_vs_iter(mean_power_vs_iter, path_to_save_folder, filename_appendix=None,
	verbose=0, tab_level=0):
	filename_prefix = 'MP'
	pm.printvm('plot_mean_power_vs_iter(). %s_%s'%(str(filename_prefix), str(filename_appendix)),
		verbose=verbose,tab_level=tab_level,verbose_threshold=100)

	x, y, y_error = [], [], []
	for this_iter, data_list in mean_power_vs_iter.items():
		x.append(this_iter)
		y.append(np.mean(data_list))
		y_error.append(np.var(data_list)**0.5)
	x, y, y_error = np.array(x), np.array(y), np.array(y_error)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x,y)
	ax.fill_between(x, y-y_error, y+y_error,color='b',alpha=0.05)
	ax.set_xlabel('iter')
	ax.set_ylabel('mean power')
	if filename_appendix is not None:
		ax.set_title(filename_appendix)

	filename = filename_prefix + '_' + filename_appendix + '.jpg'
	path_to_save_file = os.path.join(path_to_save_folder, filename)
	plt.savefig(path_to_save_file)
	plt.close()


def plot_accuracy(accuracy_vs_iter, path_to_save_folder):
	print('plot_accuracy()')
	filename = 'accuracy.jpg'
	acc_x_iter, acc_y = [], []
	for this_iter, acc in accuracy_vs_iter.items():
		acc_x_iter.append(this_iter)
		acc_y.append(acc)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(acc_x_iter, acc_y)
	ax.set_xlabel('iter')
	ax.set_ylabel('accuracy')

	path_to_save_file = os.path.join(path_to_save_folder, filename)
	plt.savefig(path_to_save_file)
	plt.close()
	print('  [END] plot_accuracy()')

def plot_loss(loss_data_by_epoch, path_to_save_folder):
	print('plot_loss()')
	filename = 'loss.jpg'

	lastest_iter = 0
	total_epoch = len(loss_data_by_epoch)
	color_scheme = np.linspace(0,1,total_epoch)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	for n_epoch in range(total_epoch):
		y = loss_data_by_epoch[n_epoch]
		x = np.array(range(1,1+len(y))) + lastest_iter
		ax.plot(x,y, color=(1-color_scheme[n_epoch],0,color_scheme[n_epoch]))
		lastest_iter = x[-1]
		ax.set_xlabel('iter')
		ax.set_ylabel('loss')
	path_to_save_file = os.path.join(path_to_save_folder, filename)
	plt.savefig(path_to_save_file)
	print('  [END] plot_loss()')

#################################################################
# Utilities
#################################################################

def print_processed_datapoint_vs_iter(processed_datapoint_vs_iter,verbose=250, tab_level=0):
	"""
    this_iter  | n-th    | data_name                        | type
    3750       | 0       | pred_is_correct                  | <class 'bool'>
               | 0       | raw_prediction                   | <class 'numpy.ndarray'>
               | 0       | prediction                       | <class 'numpy.int64'>
               | 0       | ground_truth                     | <class 'numpy.int64'>
               | 0       | meanpower_by_layer_name          | <class 'dict'>
               |         |   fn3                            |   <class 'dict'>
               |         |     positive_mean_power          |     <class 'numpy.float32'>
               |         |     negative_mean_power          |     <class 'numpy.float32'>
               |         |   fn2                            |   <class 'dict'>
               |         |   fn1                            |   <class 'dict'>
               |         |   convb_6                        |   <class 'dict'>
               |         |   convb_5                        |   <class 'dict'>
               |         |   convb_4                        |   <class 'dict'>
               |         |   convb_3                        |   <class 'dict'>
               |         |   convb_2                        |   <class 'dict'>
               |         |   convb_1                        |   <class 'dict'>
               | 1       | pred_is_correct                  | <class 'bool'>
               | 1       | raw_prediction                   | <class 'numpy.ndarray'>
               | 1       | prediction                       | <class 'numpy.int64'>
               | 1       | ground_truth                     | <class 'numpy.int64'>
               | 1       | meanpower_by_layer_name          | <class 'dict'>
               |         |   fn3                            |   <class 'dict'>
	"""
	pm.printvm('%-10s | %-7s | %-32s | %s'%(str('this_iter'), str('n-th'), str('data_name'),str('type')), 
		verbose=verbose, tab_level=tab_level+1, verbose_threshold=250)
	for this_iter, list_of_processed_data in processed_datapoint_vs_iter.items():
		this_iter0 = this_iter
		for i, processed_data in enumerate(list_of_processed_data):
			# displays
			for data_name, this_data in processed_data.items():
				if i == 0 or i==1:
					i0 = i
					pm.printvm('%-10s | %-7s | %-32s | %s'%(str(this_iter0),str(i0), str(data_name), str(type(this_data))), 
						verbose=verbose, tab_level=tab_level+1, verbose_threshold=100)
					this_iter0, i0 = '', ''
					if isinstance(this_data, dict):
						sub_count = 0
						for subdata_name, this_subdata in this_data.items():
							pm.printvm('%-10s | %-7s |   %-30s |   %s'%(str(''),str(''), str(subdata_name), str(type(this_subdata))), 
								verbose=verbose, tab_level=tab_level+1, verbose_threshold=100)
							if isinstance(this_subdata,dict) and sub_count==0:
								sub_count += 1
								for subsubdata_name, this_subsubdata in this_subdata.items():
									pm.printvm('%-10s | %-7s |     %-28s |     %s'%(str(''),str(''), str(subsubdata_name), 
										str(type(this_subsubdata))), verbose=verbose, tab_level=tab_level+1, verbose_threshold=100)													
			if i==2:
				pm.printvm('%-10s'%('...'), verbose=verbose, tab_level=tab_level, verbose_threshold=100)

def get_path_to_save_folder(state_tracker):
	path_to_save_folder = os.path.join(state_tracker.series_folder_path,'Image_drivethru_visual0001')
	if not os.path.exists(path_to_save_folder):
		os.mkdir(path_to_save_folder)
	print('path_to_save_folder:\n  %s'%(str(path_to_save_folder)))
	return path_to_save_folder