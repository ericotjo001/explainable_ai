from utils.utils import *
import models.networks_LRP as nlrp
import models.networks_utils as nutils


class ConvBlock2D_001(nn.Module):
	# conv + batch norm + LeakyReLu
	def __init__(self, in_channel, out_channel, kernel_size, 
		stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
		layer_name='convb', is_first_layer=False):
		super(ConvBlock2D_001, self).__init__()

		self.layer_name = layer_name
		self.is_first_layer = is_first_layer
		
		conv = nlrp.Conv2d_LRP(in_channel, out_channel, kernel_size, 
			padding=padding, stride=stride, dilation=dilation,
			is_first_layer=is_first_layer)
		bn = nn.BatchNorm2d(out_channel)
		act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

		setattr(self, str(self.layer_name) + '_conv', conv)
		setattr(self, str(self.layer_name) + '_bn' , bn)
		setattr(self, str(self.layer_name) + '_act', act)
		
	def forward(self, x):
		x = getattr(self, str(self.layer_name) + '_conv')(x)
		x = getattr(self, str(self.layer_name) + '_bn' )(x)
		x = getattr(self, str(self.layer_name) + '_act')(x)
		return x
	
	def forward_lrp(self, x):
		self.X = x.data
		x = getattr(self, str(self.layer_name) + '_conv').forward_lrp(x)
		# x = getattr(self, str(self.layer_name) + '_bn' )(x) # this layer not LRP-ed
		# x = getattr(self, str(self.layer_name) + '_act')(x) # this layer not LRP-ed
		return x

	def relprop(self, R, mode='relprop1'):
		if mode == 'relprop1':
			if not self.is_first_layer: R = getattr(self, str(self.layer_name) + '_conv').relprop1(R)
			else: R = getattr(self, str(self.layer_name) + '_conv').relprop1_zB(R)
		elif mode == 'relprop2':
			if not self.is_first_layer: R = getattr(self, str(self.layer_name) + '_conv').relprop2(R)
			else: R = getattr(self, str(self.layer_name) + '_conv').relprop2_zB(R)
		else:
			raise RuntimeError('Invalid Mode.')
		return R

	def relprop_debug(self, R, mode='relprop1_debug', tab_level=0, verbose=250):
		"""
		batch norm and activation layer are not LRP-ed.
		"""
		if mode == 'relprop1_debug':
			# R = getattr(self, str(self.layer_name) + '_act').relprop1(R) # this layer not LRP-ed
			# R = getattr(self, str(self.layer_name) + '_bn').relprop1(R) # this layer not LRP-ed
			if not self.is_first_layer:
				R = getattr(self, str(self.layer_name) + '_conv').relprop1_debug(R, tab_level=tab_level, verbose=verbose)
			else:
				R = getattr(self, str(self.layer_name) + '_conv').relprop1_zB_debug(R, tab_level=tab_level, verbose=verbose)	
		elif mode == 'relprop2_debug':
			# R = getattr(self, str(self.layer_name) + '_act').relprop2(R) # this layer not LRP-ed
			# R = getattr(self, str(self.layer_name) + '_bn').relprop2(R) # this layer not LRP-ed
			if not self.is_first_layer:
				R = getattr(self, str(self.layer_name) + '_conv').relprop2_debug(R, tab_level=tab_level, verbose=verbose)
			else:
				R = getattr(self, str(self.layer_name) + '_conv').relprop2_zB_debug(R, tab_level=tab_level, verbose=verbose)	

		else:
			raise RuntimeError('Invalid Mode.')
		return R