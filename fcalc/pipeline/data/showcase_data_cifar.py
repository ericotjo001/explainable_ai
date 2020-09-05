from utils.utils import *
from pipeline.data.load_data_cifar import load_cifar_0001

def showcase_cifar10_0001(config_data, verbose=250):
	print('showcase_cifar10_0001()')
	trainloader = load_cifar_0001(config_data,train=True, batch_size=None, shuffle=True, verbose=verbose)
	testloader = load_cifar_0001(config_data, train=False, batch_size=1, shuffle=False, verbose=verbose)

	print('\ntrain')
	X = []
	for i, data in enumerate(trainloader,0):
		if i>=10: break
		x,y = data
		pm.printvm('%s, %s'%(str(x.shape),str(y)))
		x1 = x.detach().numpy()
		for j in range(len(x1)):
			X.append(x1[j])

	print('\ntest')
	for i, data in enumerate(testloader,0):
		if i>=10: break
		x,y = data
		pm.printvm('%s, %s'%(str(x.shape),str(y)))
	
	X = np.concatenate(X[:10], axis=2)
	plt.figure()
	plt.imshow(np.transpose(X,(1,2,0)))
	plt.show()
	"""

	train
	torch.Size([4, 3, 32, 32]), tensor([6, 4, 0, 2])
	torch.Size([4, 3, 32, 32]), tensor([2, 6, 0, 5])
	torch.Size([4, 3, 32, 32]), tensor([4, 8, 7, 3])
	torch.Size([4, 3, 32, 32]), tensor([5, 7, 4, 0])
	torch.Size([4, 3, 32, 32]), tensor([2, 9, 5, 8])
	torch.Size([4, 3, 32, 32]), tensor([7, 6, 6, 9])
	torch.Size([4, 3, 32, 32]), tensor([0, 5, 8, 9])
	torch.Size([4, 3, 32, 32]), tensor([8, 8, 4, 6])
	torch.Size([4, 3, 32, 32]), tensor([2, 7, 0, 9])
	torch.Size([4, 3, 32, 32]), tensor([2, 0, 9, 5])

	test
	torch.Size([1, 3, 32, 32]), tensor([3])
	torch.Size([1, 3, 32, 32]), tensor([8])
	torch.Size([1, 3, 32, 32]), tensor([8])
	torch.Size([1, 3, 32, 32]), tensor([0])
	torch.Size([1, 3, 32, 32]), tensor([6])
	torch.Size([1, 3, 32, 32]), tensor([6])
	torch.Size([1, 3, 32, 32]), tensor([1])
	torch.Size([1, 3, 32, 32]), tensor([6])
	torch.Size([1, 3, 32, 32]), tensor([3])
	torch.Size([1, 3, 32, 32]), tensor([1])

	"""

def showcase_cifar10_0002(config_data, verbose=250):
	print('showcase_cifar10_0002()')
	from pipeline.data.load_data_cifar import AutoReloaderTestCIFAR

	au = AutoReloaderTestCIFAR(config_data, shuffle=False)

	for i in range(au.n_test):
		x,y = au.fetch_next_item()
		x1 = x.clone().detach().cpu().numpy()
		pm.print_in_loop('x.shape: %s, x max: %s , x min: %s , y:%s'%(
			str(x1.shape),
			str(np.max(x1)),
			str(np.min(x1)),
			str(y.clone().detach().cpu().numpy())), 
			i, au.n_test, first=5, last=2,
			tab_level=1, verbose=verbose, verbose_threshold=250)

	for i in range(10):
		x,y = au.fetch_next_item()
		x1 = x.clone().detach().cpu().numpy()
		pm.print_in_loop('x.shape: %s, x max: %s , x min: %s , y:%s'%(
			str(x1.shape),
			str(np.max(x1)),
			str(np.min(x1)),
			str(y.clone().detach().cpu().numpy())), 
			i, au.n_test, first=5, last=2,
			tab_level=1, verbose=verbose, verbose_threshold=250)
	"""
	load_cifar_0001()
	Files already downloaded and verified
	  x.shape: (4, 3, 32, 32), x max: 1.0 , x min: 0.0 , y:[3 8 8 0]
	  x.shape: (4, 3, 32, 32), x max: 1.0 , x min: 0.0 , y:[6 6 1 6]
	  x.shape: (4, 3, 32, 32), x max: 1.0 , x min: 0.023529412 , y:[3 1 0 9]
	  x.shape: (4, 3, 32, 32), x max: 1.0 , x min: 0.0 , y:[5 7 9 8]
	  x.shape: (4, 3, 32, 32), x max: 1.0 , x min: 0.0 , y:[5 7 8 6]
	  ...
	  x.shape: (4, 3, 32, 32), x max: 1.0 , x min: 0.0 , y:[3 5 3 8]
	  x.shape: (4, 3, 32, 32), x max: 0.99215686 , x min: 0.003921569 , y:[3 5 1 7]


	**AutoReloaderTestCIFAR(). Reloading test data**


	load_cifar_0001()
	Files already downloaded and verified
	  x.shape: (4, 3, 32, 32), x max: 1.0 , x min: 0.0 , y:[3 8 8 0]
	  x.shape: (4, 3, 32, 32), x max: 1.0 , x min: 0.0 , y:[6 6 1 6]
	  x.shape: (4, 3, 32, 32), x max: 1.0 , x min: 0.023529412 , y:[3 1 0 9]
	  x.shape: (4, 3, 32, 32), x max: 1.0 , x min: 0.0 , y:[5 7 9 8]
	  x.shape: (4, 3, 32, 32), x max: 1.0 , x min: 0.0 , y:[5 7 8 6]
	  ...
	"""

def showcase_cifar10_0003(config_data, verbose=250):
	print('showcase_cifar10_0003()')
	
	config_data['data_from_torch']['cifar']['resize'] = (128,128) # set to None for original size
	
	trainloader = load_cifar_0001(config_data,train=True, batch_size=None, shuffle=True, verbose=verbose)
	testloader = load_cifar_0001(config_data, train=False, batch_size=1, shuffle=False, verbose=verbose)

	print('\ntrain')
	X = []
	for i, data in enumerate(trainloader,0):
		if i>=3: break
		x,y = data
		pm.printvm('%s, %s'%(str(x.shape),str(y)))
		x1 = x.detach().numpy()
		for j in range(len(x1)):
			X.append(x1[j])

	# print('\ntest')
	# for i, data in enumerate(testloader,0):
	# 	if i>=10: break
	# 	x,y = data
	# 	pm.printvm('%s, %s'%(str(x.shape),str(y)))
	
	X = np.concatenate(X[:3], axis=2)
	plt.figure()
	plt.imshow(np.transpose(X,(1,2,0)))
	plt.show()