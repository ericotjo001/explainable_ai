from utils.utils import *

class VisualDataExtract(object):
	"""
	Prototype developed for usage in
		visual/drivethru_visual_smallnet_mnist_functions.py
	"""
	def __init__(self,):
		super(VisualDataExtract, self).__init__()

	def extract_data_by_iter_by_sample_list_0001(self, processed_datapoint_vs_iter, LAYER_NAME, data_mode,
		verbose=0, tab_level=0):
		"""
		Structure of the input prototype "processed_datapoint_vs_iter expected"
			processed_datapoint_vs_iter = {
				iter : list_of_processed_data
			}
			list_of_processed_data = [
				processed_data = {
					pred_is_correct : bool
					raw_prediction : np.array
					prediction : int
					ground_truth : int
					meanpower_by_layer_name : {
						layer_name: {
							positive_mean_power: float
							negative_mean_power: float
						}
					}

				}
			]

		output:
			data_vs_iter_this_layer = {
				this_iter = [data1, data2, ...]
			}
		"""

		pm.printvm('VisualDataExtract(). extract_data_by_iter_by_sample_list_0001().'%(),
			verbose=verbose,tab_level=tab_level+1, verbose_threshold=250)		
		pm.printvm('LAYER_NAME:%s'%(str(LAYER_NAME)),
			verbose=verbose,tab_level=tab_level+1, verbose_threshold=250)	

		data_vs_iter_this_layer = {} # { this_iter: list_of_data } list over all sampled data points
		
		for this_iter, list_of_processed_data in processed_datapoint_vs_iter.items():
			for i, processed_data in enumerate(list_of_processed_data):
				pred_is_correct = processed_data['pred_is_correct']
				raw_prediction = processed_data['raw_prediction']
				prediction = processed_data['prediction']
				ground_truth = processed_data['ground_truth']
				meanpower_by_layer_name = processed_data['meanpower_by_layer_name']
				
				if this_iter not in data_vs_iter_this_layer:
					data_vs_iter_this_layer[this_iter] = []

				if data_mode == 'positive_mean_power' or data_mode == 'negative_mean_power':
					data_vs_iter_this_layer[this_iter].append(meanpower_by_layer_name[LAYER_NAME][data_mode])
				elif data_mode == 'average_mean_power':
					pmp = meanpower_by_layer_name[LAYER_NAME]['positive_mean_power']
					nmp = meanpower_by_layer_name[LAYER_NAME]['negative_mean_power']
					data_vs_iter_this_layer[this_iter].append(0.5*(pmp+nmp))

		pm.printvm('%16s | %16s'%(str('this_iter'), str('len(data_list)')),
			verbose=verbose,tab_level=tab_level+1, verbose_threshold=250)
		for this_iter, data_list in data_vs_iter_this_layer.items():
			pm.printvm('%16s | %16s'%(str(this_iter), str(len(data_list))),
				verbose=verbose,tab_level=tab_level+1, verbose_threshold=250)
		return data_vs_iter_this_layer

	def extract_single_data_from_groundtruth_agg_vs_iter(self, groundtruth_agg_vs_iter, 
		this_iter=None, 
		ground_truth=None,
		is_correct=None,
		layer_name=None,
		data_name=None,
		tab_level=0,
		verbose=0):

		if this_iter is None or (this_iter not in groundtruth_agg_vs_iter):
			pm.printvm('extract_single_data(). available this_iter:', tab_level=tab_level, verbose=verbose, verbose_threshold=100)
			iter_list = []
			for this_iter in groundtruth_agg_vs_iter:
				pm.printvm('%s'%(str(this_iter)), tab_level=tab_level+1, verbose=verbose, verbose_threshold=10)
				iter_list.append(this_iter)
			return iter_list
		else:
			current_data = groundtruth_agg_vs_iter[this_iter]
			
		if ground_truth is None or (ground_truth not in current_data):
			pm.printvm('extract_single_data(). available ground_truth in current_data:', tab_level=tab_level, verbose=verbose, verbose_threshold=100)
			for ground_truth in current_data:
				pm.printvm('%s'%(str(ground_truth)), tab_level=tab_level+1, verbose=verbose, verbose_threshold=10)
			return None
		else:
			current_data = current_data[ground_truth]

		if is_correct is None or (is_correct not in current_data):
			pm.printvm('extract_single_data(). available is_correct in current_data:', tab_level=tab_level, verbose=verbose, verbose_threshold=100)
			for is_correct in current_data:
				pm.printvm('%s'%(str(is_correct)), tab_level=tab_level+1, verbose=verbose, verbose_threshold=10)
			return None
		else:
			current_data = current_data[is_correct]

		if layer_name is None or (layer_name not in current_data):
			pm.printvm('extract_single_data(). available layer_name in current_data:', tab_level=tab_level, verbose=verbose, verbose_threshold=100)
			for layer_name in current_data:
				pm.printvm('%s'%(str(layer_name)), tab_level=tab_level+1, verbose=verbose, verbose_threshold=10)
			return None
		else:
			current_data = current_data[layer_name]

		if data_name is None:
			pm.printvm('extract_single_data(). available data_name in current_data:', tab_level=tab_level, verbose=verbose, verbose_threshold=100)
			for data_name in current_data:
				pm.printvm('%s'%(str(data_name)), tab_level=tab_level+1, verbose=verbose, verbose_threshold=10)
			return None
		elif data_name == 'sd_of_R_highest_mean_power' or data_name == 'sd_of_R_lowest_mean_power':
			if data_name == 'sd_of_R_highest_mean_power':
				mean_data_name = 'mean_of_R_highest_mean_power'
				mean_sqr_data_name = 'mean_sqr_of_R_highest_mean_power'
			elif data_name == 'sd_of_R_lowest_mean_power':
				mean_data_name = 'mean_of_R_lowest_mean_power'
				mean_sqr_data_name = 'mean_sqr_of_R_lowest_mean_power'

			mean_data = self.extract_single_data_from_groundtruth_agg_vs_iter(groundtruth_agg_vs_iter, 
				this_iter=this_iter, 
				ground_truth=ground_truth,
				is_correct=is_correct,
				layer_name=layer_name,
				data_name=mean_data_name)
			mean_sqr_data = self.extract_single_data_from_groundtruth_agg_vs_iter(groundtruth_agg_vs_iter, 
				this_iter=this_iter, 
				ground_truth=ground_truth,
				is_correct=is_correct,
				layer_name=layer_name,
				data_name=mean_sqr_data_name)

			current_data = mean_sqr_data - (mean_data)**2
		else:
			current_data = current_data[data_name]

		if not np.all(np.isfinite(current_data)):
			print('WARNING. extract_single_data_from_groundtruth_agg_vs_iter(). Non-finite value in current_data.')
			print('  ground_truth:%s, is_correct:%s, layer_name:%s, data_name:%s'%(str(ground_truth),
				str(is_correct),str(layer_name),str(data_name)))
			print(' ',type(current_data))
			print(' ',current_data.shape)
			print('  HANDLING USING nan_to_num()')
			current_data = np.nan_to_num(current_data)
		return current_data

