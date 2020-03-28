from models.networks_components import *

"""
pm is printing manager from utilsprinting_manager.py
"""


class SmallCNN(nn.Module):
	def __init__(self, IMG_SIZE=(28, 28), INPUT_CHANNEL_SIZE = 1,
		relprop_mode='relprop1', set_default_params=True, verbose=0, tab_level=0):
		super(SmallCNN, self).__init__()
		pm.printvm('SmallCNN(). Initializing!', 
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)

		self.IMG_SIZE = IMG_SIZE
		self.INPUT_CHANNEL_SIZE = INPUT_CHANNEL_SIZE
		self.relprop_mode = relprop_mode
		self.verbose = verbose
		self.tab_level = tab_level


		# for mnist default 270 578 params
		# for cifar default 328 610 params
		self.param_args = {
			'cv1': (INPUT_CHANNEL_SIZE,24,3),
			'cv2': (24,48, 3),
			'cv3': (48,64, 3),
			'cv4': (64,48, 3),
			'cv5': (48,24, 3),
			'cv6': (24,24, 3),
			'fn1': (24,10, self.IMG_SIZE),
			'fn2': (10,10, 1),
			'fn3': (10,10, 1) # adjusts this to the number of output classes
		}

		self.param_kwargs = {
			'cv1': {'padding':1,'layer_name':'convb_1', 'is_first_layer':{'min':0.0, 'max': 1.0}},
			'cv2': {'padding':1,'layer_name':'convb_2', },	
			'cv3': {'padding':1,'layer_name':'convb_3', },
			'cv4': {'padding':1,'layer_name':'convb_4', },
			'cv5': {'padding':1,'layer_name':'convb_5', },
			'cv6': {'padding':1,'layer_name':'convb_6', },
			'fn1': {'padding':0,'layer_name':'fn_1', },
			'fn2': {'padding':0,'layer_name':'fn_2', },
			'fn3': {'padding':0,'layer_name':'fn_3', },
		}
		if set_default_params:
			self.set_params()
			
	def set_params(self):
		# in_channels, out_channels, kernel_size
		p = self.param_args
		pk = self.param_kwargs

		self.convb_1 = ConvBlock2D_001(*p['cv1'],**pk['cv1'])
		self.convb_2 = ConvBlock2D_001(*p['cv2'],**pk['cv2'])
		self.convb_3 = ConvBlock2D_001(*p['cv3'],**pk['cv3'])
		self.convb_4 = ConvBlock2D_001(*p['cv4'],**pk['cv4'])
		self.convb_5 = ConvBlock2D_001(*p['cv5'],**pk['cv5'])
		self.convb_6 = ConvBlock2D_001(*p['cv6'],**pk['cv6'])

		self.fn1 = ConvBlock2D_001(*p['fn1'],**pk['fn1'])
		self.fn2 = ConvBlock2D_001(*p['fn2'],**pk['fn2'])
		self.fn3 = ConvBlock2D_001(*p['fn3'],**pk['fn3'])
	
		for x in self.modules(): x = torch.nn.DataParallel(x, device_ids=range(torch.cuda.device_count()))
		self._init_weight()
		nutils.count_parameters(self, print_param=False, tab_level=self.tab_level+1, verbose=self.verbose)

	def forward(self, x):
		x = self.convb_1(x)
		x = self.convb_2(x)
		x = self.convb_3(x)
		x = self.convb_4(x)
		x = self.convb_5(x)
		x = self.convb_6(x)

		x = self.fn1(x)
		x = self.fn2(x)
		x = self.fn3(x)
		return x

	def forward_lrp(self, x):
		x = self.convb_1.forward_lrp(x)
		x = self.convb_2.forward_lrp(x)
		x = self.convb_3.forward_lrp(x)
		x = self.convb_4.forward_lrp(x)
		x = self.convb_5.forward_lrp(x)
		x = self.convb_6.forward_lrp(x)

		x = self.fn1.forward_lrp(x)
		x = self.fn2.forward_lrp(x)
		x = self.fn3.forward_lrp(x)
		return x

	def relprop(self, R):
		relprop_mode = self.relprop_mode 
		R = self.fn3.relprop(R, mode=relprop_mode)
		R = self.fn2.relprop(R, mode=relprop_mode)
		R = self.fn1.relprop(R, mode=relprop_mode)
		R = self.convb_6.relprop(R, mode=relprop_mode)
		R = self.convb_5.relprop(R, mode=relprop_mode)
		R = self.convb_4.relprop(R, mode=relprop_mode)
		R = self.convb_3.relprop(R, mode=relprop_mode)
		R = self.convb_2.relprop(R, mode=relprop_mode)
		R = self.convb_1.relprop(R, mode=relprop_mode)
		return R

	def relprop_debug(self,R, verbose=0, tab_level=0):
		relprop_mode = self.relprop_mode + '_debug'

		pm.printvm('SmallCNN().relprop_debug(). mode:%s'%(str(relprop_mode)),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)
		
		R = self.fn3.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250)
		pm.printvm('[0] R.shape:%s '%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		R = self.fn2.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250)
		R = self.fn1.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250)
		pm.printvm('[1]. R.shape:%s '%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		R = self.convb_6.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250)
		R = self.convb_5.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250)
		R = self.convb_4.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250)
		pm.printvm('[2]. R.shape:%s '%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		R = self.convb_3.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250)
		R = self.convb_2.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250)
		R = self.convb_1.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250)
		pm.printvm('[LRP Output]. R.shape:%s '%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		return R

	def forward_debug(self, x, verbose=0, tab_level=0):
		pm.printvm('SmallCNN(). forward_debug()', 
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)

		def fast_print(label, x):
			pm.printvm('[%s] x.shape:%s'%(str(label), str(x.shape)), tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)

		x = self.convb_1(x)
		fast_print('1', x)
		x = self.convb_2(x)
		fast_print('2', x)
		x = self.convb_3(x)
		x = self.convb_4(x)
		fast_print('2.1', x)
		x = self.convb_5(x)
		x = self.convb_6(x)
		fast_print('2.2', x)

		x = self.fn1(x)
		fast_print('3', x)
		x = self.fn2(x)
		x = self.fn3(x).squeeze()
		fast_print('Output', x)
		return x

	def _init_weight(self):
		pm.printvm('SmallCNN(). _init_weight()', 
			tab_level=self.tab_level+1, verbose=self.verbose, verbose_threshold=100)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight)


def SmallCNN_custom_setting0001(net, IMG_SIZE):
	# mnist with resize (140,140)
	# 273 381 params
	INPUT_CHANNEL_SIZE = 1
	net.param_args = {
		'cv1': (INPUT_CHANNEL_SIZE,24,3),
		'cv2': (24,48, 3),
		'cv3': (48,64, 3),
		'cv4': (64,48, 3),
		'cv5': (48,24, 3),
		'cv6': (24,1, 3),
		'fn1': (1,10, IMG_SIZE),
		'fn2': (10,10, 1),
		'fn3': (10,10, 1) # adjusts this to the number of output classes
	}

	net.param_kwargs = {
		'cv1': {'padding':1,'layer_name':'convb_1', 'is_first_layer':{'min':0.0, 'max': 1.0}},
		'cv2': {'padding':1,'layer_name':'convb_2', },	
		'cv3': {'padding':1,'layer_name':'convb_3', },
		'cv4': {'padding':1,'layer_name':'convb_4', },
		'cv5': {'padding':1,'layer_name':'convb_5', },
		'cv6': {'padding':1,'layer_name':'convb_6', },
		'fn1': {'padding':0,'layer_name':'fn_1', },
		'fn2': {'padding':0,'layer_name':'fn_2', },
		'fn3': {'padding':0,'layer_name':'fn_3', },
	}
	net.set_params()	
	return net


def SmallCNN_custom_setting0002(net, IMG_SIZE):
	# cifar with resize (256, 256)
	# 358 692 params
	INPUT_CHANNEL_SIZE = 3

	half_img_size = [int(IMG_SIZE[0]/2),int(IMG_SIZE[1]/2)]
	half_img_size = np.array(half_img_size)
	net.param_args = {
		'cv1': (INPUT_CHANNEL_SIZE,24,3),
		'cv2': (24,48, 3),
		'cv3': (48,64, 3),
		'cv4': (64,48, 3),
		'cv5': (48,7, 3),
		'cv6': (7, 1, half_img_size-8),
		'fn1': (1,10, half_img_size+9),
		'fn2': (10,10, 1),
		'fn3': (10,10, 1) # adjusts this to the number of output classes
	}

	net.param_kwargs = {
		'cv1': {'padding':1,'layer_name':'convb_1', 'is_first_layer':{'min':0.0, 'max': 1.0}},
		'cv2': {'padding':1,'layer_name':'convb_2', },	
		'cv3': {'padding':1,'layer_name':'convb_3', },
		'cv4': {'padding':1,'layer_name':'convb_4', },
		'cv5': {'padding':1,'layer_name':'convb_5', },
		'cv6': {'padding':0,'layer_name':'convb_6', },
		'fn1': {'padding':0,'layer_name':'fn_1', },
		'fn2': {'padding':0,'layer_name':'fn_2', },
		'fn3': {'padding':0,'layer_name':'fn_3', },
	}
	net.set_params()	
	return net