from utils.utils import *
from models.networks_vgg import VGGLike
from utils.lrp_utils import fraction_clamp, fraction_pass, partial_amplify

class VGGLikeprobe0001(VGGLike):
	def __init__(self, set_default_params=True, relprop_mode='relprop1', INPUT_CHANNEL_SIZE = 1,
		verbose=0, tab_level=0):
		super(VGGLikeprobe0001, self).__init__(
			set_default_params=set_default_params, 
			relprop_mode=relprop_mode, 
			INPUT_CHANNEL_SIZE = INPUT_CHANNEL_SIZE,
			verbose=verbose, tab_level=tab_level)
		
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
		
	def process_drivethru_data_0001(self, lrp_package):
		return self.prober.process_drivethru_data_0001(lrp_package)

	# def process_drivethru_data_0002(self,j, lrp_package, data_agg_by_ground_truth, n_agg=100):
	# 	return self.prober.process_drivethru_data_0002(j, lrp_package, data_agg_by_ground_truth, n_agg=n_agg)

	def find_mean_power_statistics(self, c_size, this_layer):
		return self.prober.find_mean_power_statistics(c_size, this_layer)

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