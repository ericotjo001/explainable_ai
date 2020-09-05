from utils.utils import *
import utils.lrp_utils as lu

# __NOT USED FOR NOW__
# class Conv1d_LRP(nn.Conv1d):
# 	def __init__(self, in_channels, out_channels, kernel_size, 
# 		stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
# 		super(Conv1d_LRP, self).__init__(in_channels, out_channels, kernel_size,
# 			stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

class MaxPool2d_LRP(nn.MaxPool2d):
	# nn.MaxPool2d(kernel_size, 
	# 	stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
	# nn.AvgPool2d(kernel_size, 
	# 	stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
	def __init__(self, *args, **kwargs):
		super(MaxPool2d_LRP, self).__init__(*args, **kwargs)
		self.kernel_size = args[0]
		self.stride = kwargs.get('stride')
		self.padding = kwargs.get('padding')
		self.dilation = kwargs.get('dilation')

	def forward_lrp(self,x):
		y = super(MaxPool2d_LRP, self).forward(x)
		self.X = x.data
		return y 

	def relprop1(self, R):
		this_x = self.X.data.requires_grad_(True)
		avself = self.get_avgpool_self_copy() 
		Z = avself(this_x)
		S = (R/Z).data
		(Z*S).sum().backward()
		C = this_x.grad
		R = (this_x*C).data
		return R

	def relprop1_debug(self, R, tab_level=0, verbose=250):
		# in the LRP website, any maxpool is converted to avgpool before applying LRP algorithm

		this_x = self.X.clone().detach().requires_grad_(True)
		avself = self.get_avgpool_self_copy() 
		pm.printvm('MaxPool2d_LRP() .relprop1_debug()'%(),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)

		Z = avself(this_x)
		pm.printvm('[0] Z.shape:%s'%(str(Z.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		S = (R/Z).data
		pm.printvm('[1] S.shape:%s'%(str(S.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		(Z*S).sum().backward()
		C = this_x.grad
		pm.printvm('[2] C.shape:%s'%(str(C.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		R = (this_x*C).data
		pm.printvm('[Output] R.shape:%s'%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		return R

	def get_avgpool_self_copy(self):
		avself = nn.AvgPool2d(self.kernel_size, stride=self.stride, padding=self.padding)
		return avself

class Conv2d_LRP(nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size, 
		stride=1, padding=0, dilation=1, 
		groups=1, bias=True, padding_mode='zeros',
		is_first_layer=None):
		super(Conv2d_LRP, self).__init__(in_channels, out_channels, kernel_size,
			stride=stride, padding=padding, dilation=dilation, 
			groups=groups, bias=bias, padding_mode=padding_mode)

		self.is_first_layer = is_first_layer
		""" Format for is_first_layer:
		{
			'min':0.0, # min value of the input image
			'max': 1.0 # max value of the input image
		}
		"""

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.this_bias = bias
		self.padding_mode = padding_mode

		self.epsilon = 1e-9

	def forward_lrp(self, x):
		y = super(Conv2d_LRP, self).forward(x)
		self.X = x.data
		return y

	def gradprop(self, DY):
		return F.conv_transpose2d(DY, self.weight, 
			bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)

	def relprop1(self, R):
		this_x = self.X.data.requires_grad_(True)
		Z = self.forward_lrp(this_x) + self.epsilon
		S = (R/Z).data
		(Z*S).sum().backward()
		C = this_x.grad
		R = (this_x*C).data
		return R

	def relprop1_zB(self,R):
		current_device = self.weight.data.get_device()
		if current_device < 0: current_device = None 

		this_x = self.X.data.requires_grad_(True)
		lb = (this_x.data + self.is_first_layer['min']).requires_grad_(True)
		hb = (this_x.data + self.is_first_layer['max']).requires_grad_(True)

		pself = self.get_self_copy_with_positive_only_weights(current_device)
		pself.bias.data = self.bias.data
		nself = self.get_self_copy_with_negative_only_weights(current_device)
		nself.bias.data = self.bias.data

		Z = self.forward_lrp(this_x) + self.epsilon\
			- pself.forward_lrp(lb) \
			- nself.forward_lrp(hb)
		
		S = (R/Z).data

		(Z*S).sum().backward()
		C, Cp, Cm = this_x.grad, lb.grad, hb.grad
		R = (this_x*C + lb*Cp + hb*Cm).data
		return R

	def relprop1_debug(self, R, tab_level=0, verbose=250):
		pm.printvm('Conv2d_LRP().relprop1_debug:%s'%(str()),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)
		this_x = self.X.clone().detach().requires_grad_(True)
		Z = self.forward_lrp(this_x) + self.epsilon
		pm.printvm('[0] Z.shape:%s'%(str(Z.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		S = (R/Z).data
		pm.printvm('[1] S.shape:%s'%(str(S.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		(Z*S).sum().backward()
		C = this_x.grad
		pm.printvm('[2] C.shape:%s'%(str(C.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		R = (this_x*C).data
		pm.printvm('[Output] R.shape:%s'%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		return R

	def relprop1_zB_debug(self,R, tab_level=0, verbose=250):
		pm.printvm('Conv2d_LRP().relprop1_zB_debug:%s'%(str()),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)
		current_device = self.weight.data.get_device()
		if current_device < 0: current_device = None 

		this_x = self.X.clone().detach().requires_grad_(True)
		lb = (this_x.data + self.is_first_layer['min']).requires_grad_(True)
		hb = (this_x.data + self.is_first_layer['max']).requires_grad_(True)

		pself = self.get_self_copy_with_positive_only_weights(current_device)
		pself.bias.data = self.bias.data
		nself = self.get_self_copy_with_negative_only_weights(current_device)
		nself.bias.data = self.bias.data
		# here we do not zero the bias

		Z = self.forward_lrp(this_x) + self.epsilon\
			- pself.forward_lrp(lb) \
			- nself.forward_lrp(hb)
		pm.printvm('[0] Z.shape:%s'%(str(Z.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		
		S = (R/Z).data
		pm.printvm('[1] S.shape:%s'%(str(S.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)

		(Z*S).sum().backward()
		C, Cp, Cm = this_x.grad, lb.grad, hb.grad
		R = (this_x*C + lb*Cp + hb*Cm).data
		pm.printvm('[Output First Layer] R.shape:%s'%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		return R
		
	def relprop2(self, R):
		current_device = self.weight.data.get_device()
		if current_device < 0: current_device = None 

		this_x = self.X.clone().detach().requires_grad_(True)

		pself = self.get_self_copy_with_positive_only_weights(current_device)
		pself.bias.data = pself.bias* 0

		Z = pself.forward_lrp(this_x)
		signed_small_Z = lu.find_signed_small_x(Z, self.epsilon)
		Z = Z + self.epsilon*signed_small_Z

		S = R/Z
		S = S.to(device=current_device)
		
		C = pself.gradprop(S)	

		sH, sW = self.get_resize_initial_indices_2D(C, this_x)
		R = (this_x*C[:,:,sH:sH+this_x.shape[2],sW:sW+this_x.shape[3]])
		return R
		
	def relprop2_debug(self, R, tab_level=0, verbose=250):
		pm.printvm('Conv2d_LRP(). relprop2_debug()'%(),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)
		current_device = self.weight.data.get_device()
		if current_device < 0: current_device = None 

		this_x = self.X.clone().detach().requires_grad_(True)

		pself = self.get_self_copy_with_positive_only_weights(current_device)
		pself.bias.data = pself.bias* 0

		Z = pself.forward_lrp(this_x)
		signed_small_Z = lu.find_signed_small_x(Z, self.epsilon)
		Z = Z + self.epsilon*signed_small_Z
		pm.printvm('[0] Z.shape:%s R.shape:%s'%(str(Z.shape),str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		##################################################
		nan_check = (torch.sum(torch.isnan(Z))>0).clone().detach().cpu().numpy()
		pm.printvm('nan_check:%s max Z:%s min Z:%s'%(str(nan_check),
			str(torch.max(Z).item()),str(torch.min(Z).item())),
			tab_level=tab_level+2, verbose=verbose, verbose_threshold=100)
		##################################################

		S = R/Z
		S = S.to(device=current_device)
		pm.printvm('[1] S.shape:%s'%(str(S.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		
		C = pself.gradprop(S)	
		pm.printvm('[2] C.shape:%s'%(str(C.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)

		sH, sW = self.get_resize_initial_indices_2D(C, this_x)
		R = (this_x*C[:,:,sH:sH+this_x.shape[2],sW:sW+this_x.shape[3]])
		pm.printvm('[Output] R.shape:%s'%(str(R.shape)),
			tab_level=tab_level+1, verbose=verbose, verbose_threshold=100)
		##################################################
		nan_check = (torch.sum(torch.isnan(R))>0).clone().detach().cpu().numpy()
		pm.printvm('nan_check:%s max R:%s min R:%s'%(str(nan_check),
			str(torch.max(R).item()),str(torch.min(R).item())),
			tab_level=tab_level+2, verbose=verbose, verbose_threshold=100)
		##################################################
		return R

	def relprop2_zB(self, R):
		current_device = self.weight.data.get_device()
		if current_device < 0: current_device = None 
		
		pself = self.get_self_copy_with_positive_only_weights(current_device)
		nself = self.get_self_copy_with_negative_only_weights(current_device)		
		iself = self.get_self_copy(current_device)
		pself.bias.data = pself.bias* 0
		nself.bias.data = nself.bias* 0
		iself.bias.data = iself.bias* 0
		X, L, H = self.X.clone(), self.X*0 + self.is_first_layer['min'] , \
			self.X*0 + self.is_first_layer['max']

		Z = iself.forward_lrp(X) + self.epsilon\
			- pself.forward_lrp(L) \
			- nself.forward_lrp(H)

		S = R/Z
		iC,pC,nC = iself.gradprop(S), pself.gradprop(S), nself.gradprop(S)
		sH, sW = self.get_resize_initial_indices_2D(iC, X)


		R = X*iC[:,:,sH:sH+X.shape[2],sW:sW+X.shape[3]] - \
			L*pC[:,:,sH:sH+X.shape[2],sW:sW+X.shape[3]] - \
			H*nC[:,:,sH:sH+X.shape[2],sW:sW+X.shape[3]]

		return R

	def relprop2_zB_debug(self, R, tab_level=0, verbose=250):
		pm.printvm('Conv2d_LRP(). relprop2_zB_debug()'%(),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)
		current_device = self.weight.data.get_device()
		if current_device < 0: current_device = None 
		
		pself = self.get_self_copy_with_positive_only_weights(current_device)
		nself = self.get_self_copy_with_negative_only_weights(current_device)		
		iself = self.get_self_copy(current_device)
		pself.bias.data = pself.bias* 0
		nself.bias.data = nself.bias* 0
		iself.bias.data = iself.bias* 0
		X, L, H = self.X.clone(), self.X*0 + self.is_first_layer['min'] , \
			self.X*0 + self.is_first_layer['max']

		pm.printvm('[-1] X.shape:%s'%(str(X.shape)),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)

		Z = iself.forward_lrp(X) + self.epsilon\
			- pself.forward_lrp(L) \
			- nself.forward_lrp(H)
		pm.printvm('[0] Z.shape:%s'%(str(Z.shape)),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)

		S = R/Z
		pm.printvm('[1] S.shape:%s'%(str(S.shape)),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)

		iC,pC,nC = iself.gradprop(S), pself.gradprop(S), nself.gradprop(S)
		pm.printvm('[2] iC.shape:%s pC.shape:%s nC.shape:%s'%(str(iC.shape),
			str(pC.shape),str(nC.shape)),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)

		sH, sW = self.get_resize_initial_indices_2D(iC, X)


		R = X*iC[:,:,sH:sH+X.shape[2],sW:sW+X.shape[3]] - \
			L*pC[:,:,sH:sH+X.shape[2],sW:sW+X.shape[3]] - \
			H*nC[:,:,sH:sH+X.shape[2],sW:sW+X.shape[3]]
		pm.printvm('[Output First Layer] R.shape:%s'%(str(R.shape)),
			tab_level=tab_level, verbose=verbose, verbose_threshold=100)
		return R

	def get_self_copy(self, current_device):
		iself = Conv2d_LRP(self.in_channels, self.out_channels, self.kernel_size,
			stride=self.stride, padding=self.padding, dilation=self.dilation, 
			groups=self.groups, bias=self.this_bias, 
			padding_mode=self.padding_mode).to(device=current_device)
		iself.weight.data = self.weight.data
		return iself

	def get_self_copy_with_positive_only_weights(self, current_device):
		pself = Conv2d_LRP(self.in_channels, self.out_channels, self.kernel_size,
			stride=self.stride, padding=self.padding, dilation=self.dilation, 
			groups=self.groups, bias=self.this_bias, 
			padding_mode=self.padding_mode).to(device=current_device)
		pself.weight.data = self.weight.data*(self.weight.data>0).to(self.weight.data.dtype)
		return pself

	def get_self_copy_with_negative_only_weights(self, current_device):
		nself = Conv2d_LRP(self.in_channels, self.out_channels, self.kernel_size,
			stride=self.stride, padding=self.padding, dilation=self.dilation, 
			groups=self.groups, bias=self.this_bias, 
			padding_mode=self.padding_mode).to(device=current_device)
		nself.weight.data = self.weight.data*(self.weight.data<0).to(self.weight.data.dtype)
		return nself

	def get_resize_initial_indices_2D(self,larger_tensor, smaller_tensor):
		# assume (batch_size, C, H, W)
		# assume batch_size and C are the same
		sL = larger_tensor.clone().detach().cpu().numpy().shape
		ss = smaller_tensor.clone().detach().cpu().numpy().shape
		assert(sL[0]==ss[0])
		assert(sL[1]==ss[1])
		sH = int((sL[2] - ss[2])/2)
		assert(sH>=0)
		sW = int((sL[3] - ss[3])/2)
		assert(sW>=0)
		return sH, sW
