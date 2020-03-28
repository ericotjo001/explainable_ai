from utils.utils import *
from models.networks_smallCNN import SmallCNN
from utils.lrp_utils import fraction_clamp, fraction_pass, partial_amplify

class SmallCNNprobe0001(SmallCNN):
	"""
	SmallCNNprobe0001 inherits SmallCNN.
	SmallCNN Modules:
		self.convb_1 = ConvBlock2D_001(*p['cv1'],**pk['cv1'])
		self.convb_2 = ConvBlock2D_001(*p['cv2'],**pk['cv2'])
		self.convb_3 = ConvBlock2D_001(*p['cv3'],**pk['cv3'])
		self.convb_4 = ConvBlock2D_001(*p['cv4'],**pk['cv4'])
		self.convb_5 = ConvBlock2D_001(*p['cv5'],**pk['cv5'])
		self.convb_6 = ConvBlock2D_001(*p['cv6'],**pk['cv6'])

		self.fn1 = ConvBlock2D_001(*p['fn1'],**pk['fn1'])
		self.fn2 = ConvBlock2D_001(*p['fn2'],**pk['fn2'])
		self.fn3 = ConvBlock2D_001(*p['fn3'],**pk['fn3'])
	"""
	def __init__(self, IMG_SIZE=(28, 28), INPUT_CHANNEL_SIZE = 1,
		relprop_mode='relprop1', set_default_params=True, verbose=0, tab_level=0):
		super(SmallCNNprobe0001, self).__init__(IMG_SIZE=IMG_SIZE, 
			INPUT_CHANNEL_SIZE = INPUT_CHANNEL_SIZE,
			relprop_mode=relprop_mode, 
			set_default_params=set_default_params, 
			verbose=verbose, 
			tab_level=tab_level)

		from models.networks_smallCNN_probe import ProbeClass
		self.prober = ProbeClass()

	def relprop_drivethru0001(self, x, R, y0):
		lrp_package = {}
		
		y = self.forward(x)
		lrp_package['y'] = y.data.detach().cpu().numpy()
		lrp_package['y0'] = y0.data.detach().cpu().numpy()
		lrp_package['x'] = x.data.detach().cpu().numpy()
		lrp_package['lrp_layers'] = {}

		self.forward_lrp(x)
		relprop_mode = self.relprop_mode 
		
		R = self.fn3.relprop(R, mode=relprop_mode)
		lrp_package['lrp_layers']['fn3'] = {'R': R.data.detach().cpu().numpy(),'x':self.fn3.X.data.detach().cpu().numpy()}
		R = self.fn2.relprop(R, mode=relprop_mode)
		lrp_package['lrp_layers']['fn2'] = {'R': R.data.detach().cpu().numpy(),'x':self.fn2.X.data.detach().cpu().numpy()}
		R = self.fn1.relprop(R, mode=relprop_mode)
		lrp_package['lrp_layers']['fn1'] = {'R': R.data.detach().cpu().numpy(),'x':self.fn1.X.data.detach().cpu().numpy()}

		R = self.convb_6.relprop(R, mode=relprop_mode)
		lrp_package['lrp_layers']['convb_6'] = {'R': R.data.detach().cpu().numpy(),'x':self.convb_6.X.data.detach().cpu().numpy()}		
		R = self.convb_5.relprop(R, mode=relprop_mode)
		lrp_package['lrp_layers']['convb_5'] = {'R': R.data.detach().cpu().numpy(),'x':self.convb_5.X.data.detach().cpu().numpy()}
		R = self.convb_4.relprop(R, mode=relprop_mode)
		lrp_package['lrp_layers']['convb_4'] = {'R': R.data.detach().cpu().numpy(),'x':self.convb_4.X.data.detach().cpu().numpy()}
		R = self.convb_3.relprop(R, mode=relprop_mode)
		lrp_package['lrp_layers']['convb_3'] = {'R': R.data.detach().cpu().numpy(),'x':self.convb_3.X.data.detach().cpu().numpy()}
		R = self.convb_2.relprop(R, mode=relprop_mode)
		lrp_package['lrp_layers']['convb_2'] = {'R': R.data.detach().cpu().numpy(),'x':self.convb_2.X.data.detach().cpu().numpy()}
		R = self.convb_1.relprop(R, mode=relprop_mode)
		lrp_package['lrp_layers']['convb_1'] = {'R': R.data.detach().cpu().numpy(),'x':self.convb_1.X.data.detach().cpu().numpy()}		
		lrp_package['R'] = R
		return lrp_package

	def relprop_drivethru0003(self, x, R, y0, modifier=None):	
		x1 = x.clone().detach()
		y = self.forward(x1)
		self.forward_lrp(x1)
		relprop_mode = self.relprop_mode 
		
		R = R.clone().detach()
		R = self.fn3.relprop(R, mode=relprop_mode)
		R = self.fn2.relprop(R, mode=relprop_mode)
		R = self.fn1.relprop(R, mode=relprop_mode)
		R = self.lrp_modifier(R,modifier=modifier)

		R = self.convb_6.relprop(R, mode=relprop_mode)		
		R = self.lrp_modifier(R,modifier=modifier)
		R = self.convb_5.relprop(R, mode=relprop_mode)
		R = self.lrp_modifier(R,modifier=modifier)
		R = self.convb_4.relprop(R, mode=relprop_mode)
		R = self.lrp_modifier(R,modifier=modifier)
		R = self.convb_3.relprop(R, mode=relprop_mode)
		R = self.lrp_modifier(R,modifier=modifier)
		R = self.convb_2.relprop(R, mode=relprop_mode)
		R = self.convb_1.relprop(R, mode=relprop_mode)
		return R

	def lrp_modifier(self,R,modifier=None):
		if modifier is not None:
			if modifier['mode'] == 'clamp':
				alpha1, alpha2 = modifier['alphas']
				R = fraction_clamp(R, alpha1=alpha1,alpha2=alpha2, verbose=0)
				# print('WATERMARK: CLAMPING')
			elif modifier['mode'] == 'pass':
				alpha1, alpha2 = modifier['alphas']
				R = fraction_pass(R, alpha1=alpha1,alpha2=alpha2, verbose=0)
				# print('WATERMARK: PASSING')
			elif modifier['mode'] == 'p_amp':
				alpha, amp = modifier['alphas']
				R = partial_amplify(R, alpha=alpha, amp=amp, verbose=0)
				# print('WATERMARK: AMPLIFYING')
			else:
				print('(!) WATERMARK: NOTHING? ')
		return R

	def relprop_drivethru0003_debug(self, x, R, y0, alpha1=0,alpha2=0.9):	
		print('relprop_drivethru0003_debug()')
		raise Exception('REMOVED!')
		# x1 = x.clone().detach()
		# y = self.forward(x1)
		# self.forward_lrp(x1)
		# relprop_mode = self.relprop_mode 
		
		# R = R.clone().detach()
		# R = self.fn3.relprop(R, mode=relprop_mode)
		# R = fraction_clamp(R, alpha1=alpha1,alpha2=alpha2, verbose=0)
		# R = self.fn2.relprop(R, mode=relprop_mode)
		# R = fraction_clamp(R, alpha1=alpha1,alpha2=alpha2, verbose=0)
		# R = self.fn1.relprop(R, mode=relprop_mode)
		# R = fraction_clamp(R, alpha1=alpha1,alpha2=alpha2, verbose=0)
		
		# R = self.convb_6.relprop(R, mode=relprop_mode)		
		# R = fraction_clamp(R, alpha1=alpha1,alpha2=alpha2, verbose=0)
		# R = self.convb_5.relprop(R, mode=relprop_mode)
		# R = self.convb_4.relprop(R, mode=relprop_mode)
		# R = self.convb_3.relprop(R, mode=relprop_mode)
		# R = self.convb_2.relprop(R, mode=relprop_mode)
		# R = self.convb_1.relprop(R, mode=relprop_mode)
		# print('  R.shape:%s'%(str(R.shape)))
		return R

	def relprop_drivethru0004_debug(self, x, R, y0, amplifier=1.1):	
		print('relprop_drivethru0004_debug()')
		raise Exception('REMOVED')
		return R

	def process_drivethru_data_0001(self, lrp_package):
		return self.prober.process_drivethru_data_0001(lrp_package)

	# def process_drivethru_data_0002(self,j, lrp_package, data_agg_by_ground_truth, n_agg=100):
	# 	return self.prober.process_drivethru_data_0002(j, lrp_package, data_agg_by_ground_truth, n_agg=n_agg)

	def find_mean_power_statistics(self, c_size, this_layer):
		return self.prober.find_mean_power_statistics(c_size, this_layer)

class ProbeClass(object):
	def __init__(self):
		super(ProbeClass, self).__init__()
		
		from utils.metrics_for_lrp_output import PercentileMeanPower
		self.pmp = PercentileMeanPower()
		self.pmp.mp_threshold = 0.8

	def process_drivethru_data_0001(self, lrp_package):
		processed_datapoint = {}

		raw_prediction = lrp_package['y'].reshape(-1)
		y1 = np.argmax(raw_prediction) 
		y0 = lrp_package['y0'][0] 
		pred_is_correct = int(y1)==int(y0)
		processed_datapoint['pred_is_correct'] = pred_is_correct
		processed_datapoint['raw_prediction'] = raw_prediction
		processed_datapoint['prediction'] = y1
		processed_datapoint['ground_truth'] = y0
		
		processed_datapoint['meanpower_by_layer_name'] = {}
		for layer_name, this_layer in lrp_package['lrp_layers'].items():
			positive_mean_power, negative_mean_power = self.pmp.sorted_sign_split(this_layer['R'], mp_threshold=self.pmp.mp_threshold, verbose=0)
			processed_datapoint['meanpower_by_layer_name'][layer_name] = {
				'positive_mean_power': positive_mean_power,
				'negative_mean_power': negative_mean_power
			}
		return processed_datapoint


	def find_mean_power_statistics(self, c_size, this_layer):
		max_mp, min_mp = -np.inf, np.inf
		max_mp_channel_index, min_mp_channel_index = 0, 0
		for k in range(c_size): # remark 2.1. Find the "best" mean power over the channels
			positive_mean_power, negative_mean_power = self.pmp.sorted_sign_split(this_layer['R'][0,[k],:,:], mp_threshold=self.pmp.mp_threshold, verbose=0)
			mean_power = np.max([positive_mean_power, negative_mean_power])
			if mean_power>=max_mp:
				max_mp = mean_power
				max_mp_channel_index = k
			if mean_power<= min_mp:
				min_mp = mean_power
				min_mp_channel_index = k
		mean_power_unit_statistics = {
			'max_mp': (max_mp_channel_index, max_mp),
			'min_mp': (min_mp_channel_index, min_mp)
			# _ADD OTHER METRICS YOU WANT TO STORE HERE_
		}

		if not np.all(np.isfinite([max_mp,min_mp])):
			print('WARNING. (). Non-finite value in max_mp or min_mp.')
			print('  c_size:%s'%(str(c_size)))
			print('  max_mp:%s, min_mp:%s'%(str(max_mp),str(min_mp)))

		return mean_power_unit_statistics

	def process_drivethru_data_0001_debug(self, lrp_package, verbose=250, tab_level=0):
		# 1 lrp_package is literally 1 image
		# mean power is computed over all channels (see remark 1)

		pm.printvm('process_drivethru_data_0001_debug().'%(),
			verbose=verbose,verbose_threshold=0, tab_level=tab_level)
		processed_datapoint = {}

		##########################################################
		# Basic setup
		##########################################################

		raw_prediction = lrp_package['y'].reshape(-1)
		y1 = np.argmax(raw_prediction) 
		y0 = lrp_package['y0'][0] 
		pred_is_correct = int(y1)==int(y0)
		processed_datapoint['pred_is_correct'] = pred_is_correct
		processed_datapoint['raw_prediction'] = raw_prediction
		processed_datapoint['prediction'] = y1
		processed_datapoint['ground_truth'] = y0
		pm.printvm('pred_is_correct:%s | y1 = %s | y0 = %s'%(str(pred_is_correct),str(y1),str(y0)),
			verbose=verbose,verbose_threshold=0, tab_level=tab_level+1)

		##########################################################
		# Mean power
		##########################################################
		
		processed_datapoint['meanpower_by_layer_name'] = {}
		pm.printvm('%-10s | %22s | %22s | %6s | %6s'%(str("layer_name"),str("this_layer['x'].shape"),str("this_layer['R'].shape"),
			str('+ mp'),str('- mp')),
				verbose=verbose,verbose_threshold=0, tab_level=tab_level+2)
		for layer_name, this_layer in lrp_package['lrp_layers'].items():
			# remark 1: mean power is computed over all channels. this_layer['R'] has multiple channels and the 
			#   mean power takes the mean over ALL these channels
			positive_mean_power, negative_mean_power = self.pmp.sorted_sign_split(this_layer['R'], mp_threshold=self.pmp.mp_threshold, verbose=0)
			
			processed_datapoint['meanpower_by_layer_name'][layer_name] = {
				'positive_mean_power': positive_mean_power,
				'negative_mean_power': negative_mean_power
			}
			pm.printvm('%-10s | %-22s | %-22s | %-6s | %-6s'%(str(layer_name),str(this_layer['x'].shape),str(this_layer['R'].shape),
				str(round(positive_mean_power,3)),str(round(negative_mean_power,3))),
				verbose=verbose,verbose_threshold=0, tab_level=tab_level+2)

		return processed_datapoint

	# TO DEPRECATE, by revision 1

	# def process_drivethru_data_0002(self,j, lrp_package, data_agg_by_ground_truth, n_agg=100):
	# 	# 1 lrp_package is literally 1 image
	# 	# this process computes mean power in a "finer" sense thatn process_drivethru_data_0001()
	# 	#   because we find the values in a single channel. See remark 2
	# 	# the output already average over all n_agg randomly selected data samples
	# 	raw_prediction = lrp_package['y'].reshape(-1)
	# 	y1 = int(np.argmax(raw_prediction)) 
	# 	y0 = int(lrp_package['y0'][0]) 
	# 	pred_is_correct = int(y1)==int(y0)

	# 	if y0 not in data_agg_by_ground_truth:
	# 		data_agg_by_ground_truth[y0] = {'correct_pred': {}, 'wrong_pred': {}}
		
	# 	for layer_name, this_layer in lrp_package['lrp_layers'].items():
	# 		c_size = this_layer['R'][0].shape[0]
			
	# 		pred_category = 'wrong_pred'
	# 		if pred_is_correct: pred_category = 'correct_pred'
			
	# 		if layer_name not in data_agg_by_ground_truth[y0][pred_category]:
	# 			data_agg_by_ground_truth[y0][pred_category][layer_name] = {
	# 				'mean_of_highest_mean_power': 0.,
	# 				'mean_of_lowest_mean_power': 0.,
	# 				'mean_of_R_highest_mean_power': np.zeros(shape=this_layer['R'][0][0].shape), 
	# 				'mean_of_R_lowest_mean_power': np.zeros(shape=this_layer['R'][0][0].shape),
	# 				'mean_sqr_of_R_highest_mean_power': np.zeros(shape=this_layer['R'][0][0].shape),
	# 				'mean_sqr_of_R_lowest_mean_power': np.zeros(shape=this_layer['R'][0][0].shape),
	# 			}

	# 		# remark 2. Mean power values such as max, min etc (add any other if you want) is computed for EVERY channel. See remark 2.1
	# 		mean_power_unit_statistics= self.find_mean_power_statistics(c_size, this_layer)
	# 		max_mp_channel_index = mean_power_unit_statistics['max_mp'][0]
	# 		min_mp_channel_index = mean_power_unit_statistics['min_mp'][0]

	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_of_highest_mean_power'] += mean_power_unit_statistics['max_mp'][1]/n_agg
	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_of_lowest_mean_power'] += mean_power_unit_statistics['min_mp'][1]/n_agg
	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_of_R_highest_mean_power'] += this_layer['R'][0][max_mp_channel_index]/n_agg
	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_of_R_lowest_mean_power'] += this_layer['R'][0][min_mp_channel_index]/n_agg
	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_sqr_of_R_highest_mean_power'] += (this_layer['R'][0][max_mp_channel_index]**2)/n_agg
	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_sqr_of_R_lowest_mean_power'] += (this_layer['R'][0][min_mp_channel_index]**2)/n_agg
		
	# 		for checkkey, check_value in data_agg_by_ground_truth[y0][pred_category][layer_name].items():
	# 			if not np.all(np.isfinite(check_value)):
	# 				print('WARNING. (). Non-finite value in data_agg_by_ground_truth[y0][pred_category][layer_name].')
	# 				print('  y0:%s, pred_category:%s, layer_name:%s'%(str(y0),str(pred_category),str(layer_name),))
				
	# 	return data_agg_by_ground_truth

	# def process_drivethru_data_0002_debug(self, j, lrp_package, data_agg_by_ground_truth, 
	# 	n_agg=100, verbose=250, tab_level=0):
	# 	if j==0:
	# 		pm.printvm('process_drivethru_data_0002_debug(). n_agg:%s'%(str(n_agg)),
	# 			verbose=verbose,verbose_threshold=0, tab_level=tab_level)
		
	# 	raw_prediction = lrp_package['y'].reshape(-1)
	# 	y1 = int(np.argmax(raw_prediction)) 
	# 	y0 = int(lrp_package['y0'][0]) 
	# 	pred_is_correct = int(y1)==int(y0)

	# 	if y0 not in data_agg_by_ground_truth:
	# 		data_agg_by_ground_truth[y0] = {'correct_pred': {}, 'wrong_pred': {}}
		
	# 	for layer_name, this_layer in lrp_package['lrp_layers'].items():
	# 		c_size = this_layer['R'][0].shape[0]
	# 		if j==0:
	# 			# e.g.
	# 			# layer_name:convb_5
	# 			# this_layer[R][0].shape:(48, 28, 28)
	# 			# this_layer[R][0,[0],:,:].shape:(1, 28, 28)
	# 			pm.printvm('layer_name:%s'%(str(layer_name)), verbose=verbose,verbose_threshold=0, tab_level=tab_level+1)				
	# 			pm.printvm('this_layer[R][0].shape:%s'%(str(this_layer['R'][0].shape)), verbose=verbose,verbose_threshold=0, tab_level=tab_level+2)
	# 			pm.printvm('this_layer[R][0,[0],:,:].shape:%s'%(str(this_layer['R'][0,[0],:,:].shape)), verbose=verbose,verbose_threshold=0, tab_level=tab_level+2)
	# 			pm.printvm('c_size:%s'%(str(c_size)), verbose=verbose,verbose_threshold=0, tab_level=tab_level+2)		
			
	# 		pred_category = 'wrong_pred'
	# 		if pred_is_correct: pred_category = 'correct_pred'
			
	# 		if layer_name not in data_agg_by_ground_truth[y0][pred_category]:
	# 			data_agg_by_ground_truth[y0][pred_category][layer_name] = {
	# 				'mean_of_highest_mean_power': 0.,
	# 				'mean_of_lowest_mean_power': 0.,
	# 				'mean_of_R_highest_mean_power': np.zeros(shape=this_layer['R'][0][0].shape), 
	# 				'mean_of_R_lowest_mean_power': np.zeros(shape=this_layer['R'][0][0].shape),
	# 				'mean_sqr_of_R_highest_mean_power': np.zeros(shape=this_layer['R'][0][0].shape),
	# 				'mean_sqr_of_R_lowest_mean_power': np.zeros(shape=this_layer['R'][0][0].shape),
	# 			}

	# 		mean_power_unit_statistics= self.find_mean_power_statistics(c_size, this_layer)
	# 		max_mp_channel_index = mean_power_unit_statistics['max_mp'][0]
	# 		min_mp_channel_index = mean_power_unit_statistics['min_mp'][0]

	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_of_highest_mean_power'] += mean_power_unit_statistics['max_mp'][1]/n_agg
	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_of_lowest_mean_power'] += mean_power_unit_statistics['min_mp'][1]/n_agg
	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_of_R_highest_mean_power'] += this_layer['R'][0][max_mp_channel_index]/n_agg
	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_of_R_lowest_mean_power'] += this_layer['R'][0][min_mp_channel_index]/n_agg
	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_sqr_of_R_highest_mean_power'] += (this_layer['R'][0][max_mp_channel_index]**2)/n_agg
	# 		data_agg_by_ground_truth[y0][pred_category][layer_name]['mean_sqr_of_R_lowest_mean_power'] += (this_layer['R'][0][min_mp_channel_index]**2)/n_agg
					
	# 	return data_agg_by_ground_truth		

