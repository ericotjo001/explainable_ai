from utils.utils import *

def fraction_clamp(x, alpha1=0,alpha2=0.9, verbose=0):
	""" 
	x is any tensor in the form (batch_size, ...)
	e.g.
	x = np.random.randint(-10,10,size=(3,2,5), )
	x = torch.tensor(x)

	print(x)
	y = fraction_clamp(x, alpha1=0.5,alpha2=1, verbose=0)
	print(y)
	"""
	y = x*0
	for i, x1 in enumerate(x):
		x1 = x1.clone().detach().to(torch.float)
		absmax = torch.max(torch.abs(x1.reshape(-1))).item()
		# x1 = torch.clamp(x1, )
		x1 = x1/absmax
		x1p = x1*(x1>=0).to(torch.float)
		x1m = -1*x1*(x1<0).to(torch.float)

		if verbose>=500:
			print('%s'%(str(i)))
			print(absmax*(x1p-x1m))
			print(x1p)
			print(torch.clamp(x1p,alpha1, alpha2))
			print(x1m)
			print(torch.clamp(x1m,alpha1, alpha2))
		x1p = torch.clamp(x1p,alpha1, alpha2)*(x1>=0).to(torch.float)
		x1m = torch.clamp(x1m,alpha1, alpha2)*(x1<0).to(torch.float)
		y[i] = absmax*(x1p-x1m)
	return y

def fraction_pass(x, alpha1=0,alpha2=0.9, verbose=0):
	""" 
	x is any tensor in the form (batch_size, ...)
	e.g.
	x = np.random.randint(-10,10,size=(3,2,5), )
	x = torch.tensor(x)

	print(x)
	y = fraction_pass(x, alpha1=0.5,alpha2=1, verbose=0)
	print(y)
	"""
	y = x*0
	for i, x1 in enumerate(x):
		x1 = x1.clone().detach().to(torch.float)
		absmax = torch.max(torch.abs(x1.reshape(-1))).item()
		# x1 = torch.clamp(x1, )
		x1 = x1/absmax
		x1p = x1*(x1>=0).to(torch.float)
		x1m = -1*x1*(x1<0).to(torch.float)

		if verbose>=500:
			print('%s'%(str(i)))
			print(absmax*(x1p-x1m))
			print(x1p)
			print(torch.clamp(x1p,alpha1, alpha2))
			print(x1m)
			print(torch.clamp(x1m,alpha1, alpha2))
		x1p = x1p*(x1p>=alpha1).to(torch.float)*(x1p<=alpha2).to(torch.float)
		x1m = x1m*(x1m>=alpha1).to(torch.float)*(x1m<=alpha2).to(torch.float)
		y[i] = absmax*(x1p-x1m)
	return y

def partial_amplify(x, alpha=0.5, amp=2., verbose=0):
	"""e.g.
	x = np.random.randint(-10,10,size=(3,2,10), )
	x = torch.tensor(x)

	print(x)
	y = partial_amplify(x, alpha=0.5, amp=2., verbose=0)
	print(y)
	"""
	y = x*0
	for i, x1 in enumerate(x):
		x1 = x1.clone().detach().to(torch.float)
		absmax = torch.max(torch.abs(x1.reshape(-1))).item()
		# x1 = torch.clamp(x1, )
		x1 = x1/absmax
		x1p = x1*(x1>=0).to(torch.float)
		x1m = -1*x1*(x1<0).to(torch.float)

		xp_multiplier = 1 + (amp-1)*(x1p>=alpha).to(torch.float)
		xm_multiplier = 1 + (amp-1)*(x1m>=alpha).to(torch.float)

		x1p = xp_multiplier*x1p
		x1m = xm_multiplier*x1m
		y[i] = absmax*(x1p-x1m)
	return y

def get_zero_container(x,y):
	"""
	Assume x, y are tensors of the same dimension
	but not necessarily have the same size
	for example x can be 3,4,5 and y 4,3,5
	return the size that can contain both: 4,4,5
	
	Example:
	x = torch.tensor(np.random.normal(0,1,size=(3,4,5)))
	y = torch.tensor(np.random.normal(0,1,size=(4,3,5)))
	z = get_zero_container(x,y)
	print(z.shape) # torch.Size([4, 4, 5])
	"""
	s = []
	for sx,sy in zip(x.shape,y.shape):
		s.append(np.max([sx,sy]))
	return torch.zeros(s,dtype=torch.float)

def relprop_size_adjustment_2D(Z,R):
	# assume Z, R are torch tensors (batch_size,channel, H,W)
	sZ, sR = Z.shape, R.shape
	if not torch.tensor(sZ==sR).all():
		tempR, tempZ = get_zero_container(Z,R), get_zero_container(Z,R)
		z0 = [int((x-y)/2) for x,y in zip(tempZ.shape, Z.shape)]
		r0 = [int((x-y)/2) for x,y in zip(tempR.shape, R.shape)]
		# print(z0[2],z0[2]+sZ[2])
		# print(z0[3],z0[3]+sZ[3])
		# print(r0[2],r0[2]+sR[2])
		# print(r0[3],r0[3]+sR[3])
		tempZ[z0[0]:z0[0]+sZ[0],z0[1]:z0[1]+sZ[1],z0[2]:z0[2]+sZ[2],z0[3]:z0[3]+sZ[3]] = Z.to(torch.float); # Z = tempZ
		tempR[r0[0]:r0[0]+sR[0],r0[1]:r0[1]+sR[1],r0[2]:r0[2]+sR[2],r0[3]:r0[3]+sR[3]] = R.to(torch.float) # R = tempR
	else: return Z, R

	Zdev = Z.get_device()
	Rdev = R.get_device()
	if Zdev<0: Zdev=None
	if Rdev<0: Rdev=None

	return tempZ.to(device=Zdev), tempR.to(device=Rdev)

def force_fit_size_x_to_y_2D(x, y):
	# assume x,y are both torch tensors (batch_size,channel, H,W)
	# assume size y is smaller equal to x
	sx, sy = x.shape, y.shape
	x1 = y*0
	r0 = [int((x-y)/2) for x,y in zip(x.shape, y.shape)]
	for r in r0: assert(r>=0)
	x1 = x[r0[0]:r0[0]+sy[0],r0[1]:r0[1]+sy[1],r0[2]:r0[2]+sy[2],r0[3]:r0[3]+sy[3]]
	xdev = x.get_device()
	if xdev<0: xdev = None
	return x1.to(device=xdev)

def find_signed_small_x(x, small_number):
	small_x = (torch.abs(x)<small_number).to(torch.float)
	positive_small_x = ((small_x*x)>0).to(torch.float)
	negative_small_x = ((small_x*x)<0).to(torch.float)
	signed_small_x = positive_small_x - negative_small_x
	return signed_small_x