from models.networks_components2 import *

"""
pm is printing manager from utilsprinting_manager.py
"""

class VGGLike(nn.Module):
	def __init__(self, set_default_params=True, relprop_mode='relprop1', INPUT_CHANNEL_SIZE = 1,
		verbose=0, tab_level=0):
		super(VGGLike, self).__init__()
		pm.printvm('VGGLike(). Initializing!', 
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)
		
		self.relprop_mode = relprop_mode		
		self.verbose = verbose
		self.tab_level = tab_level
		self.INPUT_CHANNEL_SIZE = INPUT_CHANNEL_SIZE

		self.param_args = {
			'cv1': (self.INPUT_CHANNEL_SIZE,64,3),
			'cv2': (64,128,3),
			'cv3': (128,256,3),
			'cv4': (256,512,3),
			'cv5': (512,512,3),

			'fn1': (512,1024, 1),
			'fn2': (1024,1024, 1),
			'fn3': (1024,10, 1)
		}

		self.param_kwargs = {
			'cv1': {'padding':2,'layer_name':'convb_1', 
				'mp_kernel_size':2, 'chain_length':2, 'is_first_layer':{'min':0.0, 'max': 1.0}},
			'cv2': {'padding':1,'layer_name':'convb_2', 'mp_kernel_size':2,'chain_length':2},
			'cv3': {'padding':1,'layer_name':'convb_3', 'mp_kernel_size':2,'chain_length':4},
			'cv4': {'padding':1,'layer_name':'convb_4', 'mp_kernel_size':2,'chain_length':4},
			'cv5': {'padding':1,'layer_name':'convb_5', 'mp_kernel_size':2,'chain_length':4},

			'fn1': {'padding':0,'layer_name':'fn_1'},
			'fn2': {'padding':0,'layer_name':'fn_2'},
			'fn3': {'padding':0,'layer_name':'fn_3'},
		}

		if set_default_params:
			self.set_params()
		nutils.count_parameters(self, print_param=False, tab_level=self.tab_level+1, verbose=verbose)

	def set_params(self):
		p = self.param_args
		pk = self.param_kwargs

		self.convb_1 = ConvChain2D_002mp(*p['cv1'],**pk['cv1'])
		self.convb_2 = ConvChain2D_002mp(*p['cv2'],**pk['cv2'])
		self.convb_3 = ConvChain2D_002mp(*p['cv3'],**pk['cv3'])
		self.convb_4 = ConvChain2D_002mp(*p['cv4'],**pk['cv4'])
		self.convb_5 = ConvChain2D_002mp(*p['cv5'],**pk['cv5'])

		self.fn1 = ConvBlock2D_002(*p['fn1'],**pk['fn1'])
		self.fn2 = ConvBlock2D_002(*p['fn2'],**pk['fn2'])
		self.fn3 = ConvBlock2D_002(*p['fn3'],**pk['fn3'])

		for x in self.modules(): x = torch.nn.DataParallel(x, device_ids=range(torch.cuda.device_count()))
		self._init_weight()


	def forward(self, x):	
		x = self.convb_1(x)
		x = self.convb_2(x)
		x = self.convb_3(x)
		x = self.convb_4(x)		
		x = self.convb_5(x)
		x = self.fn1(x)
		x = self.fn2(x)
		x = self.fn3(x)
		return x

	def forward_lrp(self,x):
		x = self.convb_1.forward_lrp(x)
		x = self.convb_2.forward_lrp(x)
		x = self.convb_3.forward_lrp(x)
		x = self.convb_4.forward_lrp(x)	
		x = self.convb_5.forward_lrp(x)

		x = self.fn1.forward_lrp(x)
		x = self.fn2.forward_lrp(x)
		x = self.fn3.forward_lrp(x)
		return x		

	def forward_debug(self, x, verbose=0, tab_level=0):
		pm.printvm('VGGLike(). forward_debug()', 
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)
		
		x = self.convb_1.forward_debug(x,tab_level=tab_level+1,verbose=verbose)
		pm.printvm('[cv1] x.shape:%s '%(str(x.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)

		x = self.convb_2.forward_debug(x,tab_level=tab_level+1,verbose=verbose)
		pm.printvm('[cv2] x.shape:%s '%(str(x.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)

		x = self.convb_3.forward_debug(x,tab_level=tab_level+1,verbose=verbose)
		pm.printvm('[cv3] x.shape:%s '%(str(x.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)

		x = self.convb_4.forward_debug(x,tab_level=tab_level+1,verbose=verbose)
		pm.printvm('[cv4] x.shape:%s '%(str(x.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		
		x = self.convb_5.forward_debug(x,tab_level=tab_level+1,verbose=verbose)
		pm.printvm('[cv5] x.shape:%s '%(str(x.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)

		# no debug version
		x = self.fn1(x)
		pm.printvm('[fn1] x.shape:%s '%(str(x.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		x = self.fn2(x)
		pm.printvm('[fn2] x.shape:%s '%(str(x.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		x = self.fn3(x)
		pm.printvm('[Output] x.shape:%s '%(str(x.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		return x

	def relprop(self, R):
		relprop_mode = self.relprop_mode
		R = self.fn3.relprop(R, mode=relprop_mode) 		
		R = self.fn2.relprop(R, mode=relprop_mode)
		R = self.fn1.relprop(R, mode=relprop_mode)
		R = self.convb_5.relprop(R, mode=relprop_mode)
		R = self.convb_4.relprop(R, mode=relprop_mode) 
		R = self.convb_3.relprop(R, mode=relprop_mode) 
		R = self.convb_2.relprop(R, mode=relprop_mode) 
		R = self.convb_1.relprop(R, mode=relprop_mode)
		return R

	def relprop_debug(self,R, verbose=0, tab_level=0):
		relprop_mode = self.relprop_mode

		pm.printvm('VGGLike().relprop_debug(). mode:%s'%(str(relprop_mode)),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)
		
		R = self.fn3.relprop(R, mode=relprop_mode) # no debug version
		pm.printvm('[fn3] R.shape:%s '%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		R = self.fn2.relprop(R, mode=relprop_mode)
		R = self.fn1.relprop(R, mode=relprop_mode)
		pm.printvm('[fn1]. R.shape:%s '%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		R = self.convb_5.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250)
		pm.printvm('[cv5]. R.shape:%s '%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		R = self.convb_4.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250) 
		pm.printvm('[cv4]. R.shape:%s '%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		R = self.convb_3.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250) 
		R = self.convb_2.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250) 
		R = self.convb_1.relprop_debug(R, mode=relprop_mode, tab_level=tab_level+2, verbose=250)
		pm.printvm('[LRP Output]. R.shape:%s '%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		return R

	def _init_weight(self):
		pm.printvm('VGGLike(). _init_weight()', 
			tab_level=self.tab_level+1, verbose=self.verbose, verbose_threshold=100)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight)