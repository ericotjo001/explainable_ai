from utils.utils import *
import models.networks_LRP as nlrp
import models.networks_utils as nutils

class ConvBlock2D_002(nn.Module):
	# conv + LeakyReLu
	"""
	Similar to models/network_components.py ConvBlock2D_001 except without bn layer. 
	We define this for convenience.
	There will be no debug version.
	"""
	def __init__(self, in_channel, out_channel, kernel_size, 
		stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
		layer_name='convb', is_first_layer=False):
		super(ConvBlock2D_002, self).__init__()

		self.layer_name = layer_name
		self.is_first_layer = is_first_layer # format is like {'min':0.0,'max':1.0} or False
		conv = nlrp.Conv2d_LRP(in_channel, out_channel, kernel_size, 
			padding=padding, stride=stride, dilation=dilation,
			is_first_layer=is_first_layer)
		act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

		setattr(self, str(self.layer_name) + '_conv', conv)
		setattr(self, str(self.layer_name) + '_act', act)	

	def forward(self, x):
		x = getattr(self, str(self.layer_name) + '_conv')(x)
		x = getattr(self, str(self.layer_name) + '_act')(x)
		return x
	
	def forward_lrp(self, x):
		self.X = x.data
		x = getattr(self, str(self.layer_name) + '_conv').forward_lrp(x)
		# no need for activation layer. We do not apply LRP to it.
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

class ConvChain2D_002(nn.Module):
	""" A chain of ConvBlock2D_002 with the same number sizes"""
	def __init__(self, in_channel, out_channel, kernel_size, chain_length=3,
		stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
		layer_name='convc', is_first_layer=False):
		super(ConvChain2D_002, self).__init__()

		self.layer_name = layer_name
		self.is_first_layer = is_first_layer # format is like {'min':0.0,'max':1.0} or False
		self.chain_length = chain_length

		for i in range(chain_length):
			very_first_layer = False
			if i == 0 and is_first_layer is not None:
				very_first_layer = is_first_layer
			if i == 0:
				current_in_channel = in_channel
			else:
				current_in_channel = out_channel
			conv = nlrp.Conv2d_LRP(current_in_channel, out_channel, kernel_size, 
				padding=padding, stride=stride, dilation=dilation,
				is_first_layer=very_first_layer)
			setattr(self, str(self.layer_name) + '_chain_' + str(i), conv)

		act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
		setattr(self, str(self.layer_name) + '_act', act)

	def forward(self, x):
		for i in range(self.chain_length):
			x = getattr(self, str(self.layer_name) + '_chain_' + str(i))(x)
		x = getattr(self, str(self.layer_name) + '_act')(x)
		return x

	def forward_lrp(self, x):
		self.X = x.data
		for i in range(self.chain_length):
			x = getattr(self, str(self.layer_name) + '_chain_' + str(i)).forward_lrp(x)
		return x

	def forward_debug(self, x, tab_level=0, verbose=250):
		for i in range(self.chain_length):
			x = getattr(self, str(self.layer_name) + '_chain_' + str(i))(x)
			pm.printvm('[c%s] x.shape:%s'%(str(i),str(x.shape)),
				tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		x = getattr(self, str(self.layer_name) + '_act')(x)
		pm.printvm('[c Act] x.shape:%s'%(str(x.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		return x

	def relprop(self, R, mode='relprop1'):
		for i in range(self.chain_length-1,-1,-1):		
			if mode == 'relprop1':
				if not self.is_first_layer or not (i==0): 
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop1(R)
				else: 
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop1_zB(R)
			elif mode == 'relprop2':
				if not self.is_first_layer or not (i==0): 
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop2(R)
				else: 
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop2_zB(R)
			else:
				raise RuntimeError('Invalid Mode.')

		return R

	def relprop_debug(self, R, mode='relprop1', tab_level=0, verbose=250):
		pm.printvm('ConvChain2D_002 .relprop_debug().mode:%s'%(str(mode)), tab_level=tab_level, verbose=verbose)
		for i in range(self.chain_length-1,-1,-1):
			pm.printvm('[chain no. %s] '%(str(i)),
				tab_level=tab_level, verbose=verbose, verbose_threshold=100)			
			if mode == 'relprop1':
				if not self.is_first_layer or not (i==0): 
					# R = no relprop for activation layer
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop1_debug(R, tab_level=tab_level+1, verbose=verbose)
				else: 
					# R = no relprop for activation layer
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop1_zB_debug(R, tab_level=tab_level+1, verbose=verbose)
			elif mode == 'relprop2':
				if not self.is_first_layer or not (i==0): 
					# R = no relprop for activation layer
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop2_debug(R, tab_level=tab_level+1, verbose=verbose)
				else: 
					# R = no relprop for activation layer
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop2_zB_debug(R, tab_level=tab_level+1, verbose=verbose)
			else:
				raise RuntimeError('Invalid Mode.')
		return R

class ConvBlock2D_002mp(ConvBlock2D_002):
	"""
	ConvBlock2D_002 followed by max pooling
	"""
	def __init__(self, in_channel, out_channel, kernel_size, mp_kernel_size=2,
		stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
		mp_stride = None, mp_padding=0, mp_dilation=1,
		layer_name='convb', is_first_layer=False):

		super(ConvBlock2D_002mp, self).__init__(in_channel, out_channel, kernel_size, 
			stride=stride, padding=padding, dilation=dilation, groups=groups, 
			bias=bias, padding_mode=padding,
			layer_name=layer_name, is_first_layer=is_first_layer)
		mp = nlrp.MaxPool2d_LRP(mp_kernel_size, stride=mp_stride, padding=mp_padding, dilation=mp_dilation)
		setattr(self, str(layer_name) + '_mp', mp)	

	def forward(self, x):
		x = getattr(self, str(self.layer_name) + '_conv')(x)
		x = getattr(self, str(self.layer_name) + '_act')(x)
		x = getattr(self, str(self.layer_name) + '_mp')(x)
		return x

	def forward_debug(self, x, tab_level=0, verbose=250):
		pm.printvm('ConvBlock2D_002mp() .forward_debug()',
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)
		x = getattr(self, str(self.layer_name) + '_conv')(x)
		pm.printvm('[0] x.shape:%s'%(str(x.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		x = getattr(self, str(self.layer_name) + '_act')(x)
		x = getattr(self, str(self.layer_name) + '_mp')(x)
		pm.printvm('[1] x.shape:%s'%(str(x.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		return x

	def forward_lrp(self, x):
		self.X = x.data
		x = getattr(self, str(self.layer_name) + '_conv').forward_lrp(x)
		x = getattr(self, str(self.layer_name) + '_mp').forward_lrp(x)
		return x

	def relprop(self, R, mode='relprop1'):
		if mode == 'relprop1':
			if not self.is_first_layer: 
				R = getattr(self, str(self.layer_name) + '_mp').relprop1(R)
				R = getattr(self, str(self.layer_name) + '_conv').relprop1(R)
			else: 
				R = getattr(self, str(self.layer_name) + '_mp').relprop1(R)
				R = getattr(self, str(self.layer_name) + '_conv').relprop1_zB(R)
		elif mode == 'relprop2':
			if not self.is_first_layer: 
				R = getattr(self, str(self.layer_name) + '_mp').relprop1(R)
				R = getattr(self, str(self.layer_name) + '_conv').relprop2(R)
			else: 
				R = getattr(self, str(self.layer_name) + '_mp').relprop1(R)
				R = getattr(self, str(self.layer_name) + '_conv').relprop2_zB(R)		
		else:
			raise RuntimeError('Invalid Mode.')
		return R

	def relprop_debug(self, R, mode='relprop1', tab_level=0, verbose=250):
		pm.printvm('ConvBlock2D_002mp().relprop_debug().mode:%s'%(str(mode)), tab_level=tab_level, verbose=verbose)
		if mode == 'relprop1':
			if not self.is_first_layer: 
				R = getattr(self, str(self.layer_name) + '_mp').relprop1_debug(R, tab_level=tab_level+1, verbose=verbose)
				# R = no relprop for activation layer
				R = getattr(self, str(self.layer_name) + '_conv').relprop1_debug(R, tab_level=tab_level+1, verbose=verbose)
			else: 
				R = getattr(self, str(self.layer_name) + '_mp').relprop1_debug(R, tab_level=tab_level+1, verbose=verbose)
				# R = no relprop for activation layer
				R = getattr(self, str(self.layer_name) + '_conv').relprop1_zB_debug(R, tab_level=tab_level+1, verbose=verbose)
		elif mode == 'relprop2':
			if not self.is_first_layer: 
				R = getattr(self, str(self.layer_name) + '_mp').relprop1_debug(R, tab_level=tab_level+1, verbose=verbose)
				# R = no relprop for activation layer
				R = getattr(self, str(self.layer_name) + '_conv').relprop2_debug(R, tab_level=tab_level+1, verbose=verbose)
			else: 
				R = getattr(self, str(self.layer_name) + '_mp').relprop1_debug(R, tab_level=tab_level+1, verbose=verbose)
				# R = no relprop for activation layer
				R = getattr(self, str(self.layer_name) + '_conv').relprop2_zB_debug(R, tab_level=tab_level+1, verbose=verbose)	

		else:
			raise RuntimeError('Invalid Mode.')
		return R

class ConvChain2D_002mp(ConvChain2D_002):
	"""ConvChain2D_002 with max pool at the end"""
	def __init__(self, in_channel, out_channel, kernel_size, chain_length=3, mp_kernel_size=2,
		stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
		mp_stride = None, mp_padding=0, mp_dilation=1,
		layer_name='convb', is_first_layer=False):

		super(ConvChain2D_002mp, self).__init__(in_channel, out_channel, kernel_size, chain_length=chain_length,
			stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
			layer_name=layer_name, is_first_layer=is_first_layer)
		mp = nlrp.MaxPool2d_LRP(mp_kernel_size, stride=mp_stride, padding=mp_padding, dilation=mp_dilation)
		setattr(self, str(layer_name) + '_mp', mp)	

	def forward(self, x):
		x = super(ConvChain2D_002mp, self).forward(x)
		x = getattr(self, str(self.layer_name) + '_mp')(x)
		return x

	def forward_debug(self, x, tab_level=0, verbose=250):
		x = super(ConvChain2D_002mp, self).forward_debug(x, tab_level=tab_level, verbose=verbose)
		x = getattr(self, str(self.layer_name) + '_mp')(x)
		return x

	def forward_lrp(self, x):
		self.X = x.data
		x = super(ConvChain2D_002mp, self).forward_lrp(x)
		x = getattr(self, str(self.layer_name) + '_mp').forward_lrp(x)
		return x

	def relprop(self, R, mode='relprop1'):
		R = getattr(self, str(self.layer_name) + '_mp').relprop1(R)
		
		for i in range(self.chain_length-1,-1,-1):		
			if mode == 'relprop1':
				if not self.is_first_layer or not (i==0): 
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop1(R)
				else:
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop1_zB(R)
			elif mode == 'relprop2':
				if not self.is_first_layer or not (i==0):
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop2(R)
				else:
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop2_zB(R)
			else:
				raise RuntimeError('Invalid Mode.')
		return R

	def relprop_debug(self, R, mode='relprop1', tab_level=0, verbose=250):
		pm.printvm('ConvChain2D_002mp() .relprop_debug().mode:%s'%(str(mode)), tab_level=tab_level, verbose=verbose)
		
		R = getattr(self, str(self.layer_name) + '_mp').relprop1_debug(R, 
			tab_level=tab_level+1, verbose=verbose)
		
		for i in range(self.chain_length-1,-1,-1):
			pm.printvm('[chain no. %s] '%(str(i)),
				tab_level=tab_level, verbose=verbose, verbose_threshold=100)			
			if mode == 'relprop1':
				if not self.is_first_layer or not (i==0): 
					# R = no relprop for activation layer
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop1_debug(R, tab_level=tab_level+1, verbose=verbose)
				else:
					# R = no relprop for activation layer
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop1_zB_debug(R, tab_level=tab_level+1, verbose=verbose)
			elif mode == 'relprop2':
				if not self.is_first_layer or not (i==0): 
					# R = no relprop for activation layer
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop2_debug(R, tab_level=tab_level+1, verbose=verbose)
				else: 
					# R = no relprop for activation layer
					R = getattr(self, str(self.layer_name) + '_chain_' + str(i)).relprop2_zB_debug(R, tab_level=tab_level+1, verbose=verbose)
			else:
				raise RuntimeError('Invalid Mode.')
		return R