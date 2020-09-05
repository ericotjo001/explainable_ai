from utils.utils import *

def showcase_mnist_0001(config_data, verbose=0):
	print('showcase_mnist_0001()')
	from pipeline.data.load_data import load_mnist_0001

	config_data['data_from_torch']['mnist']['resize'] = (64,64) # set to None for original size
	config_data['data_from_torch']['mnist']['training_mode'] = True
	dataloader = load_mnist_0001(config_data, batch_size=1, download = False, 
		shuffle=False, verbose=0)

	this_iter = iter(dataloader)
	iter_counter = 0 
	n_iter = len(this_iter)

	while iter_counter<n_iter:
		iter_counter+=1

		x,y = this_iter.next()
		x1 = x.clone().detach().cpu().numpy()
		pm.print_in_loop('x.shape: %s, x max: %s , x min: %s , y:%s'%(
			str(x1.shape),
			str(np.max(x1)),
			str(np.min(x1)),
			str(y.clone().detach().cpu().numpy())), 
			iter_counter, n_iter, first=5, last=2,
			tab_level=1, verbose=verbose, verbose_threshold=250)

	print("***Done*** iter is used up")

	try:
		this_iter.next()
	except:
		print("One more next() and error will be raised. Reached end!")


def showcase_mnist_0002(config_data, verbose=0):
	print('showcase_mnist_0002()')
	"""
	showcase_mnist_0002()
	load_mnist_0001().INTERNAL SWITCH? YES.
	  x.shape: (1, 1, 28, 28), x max: 1.0 , x min: 0.0 , y:[7]
	  x.shape: (1, 1, 28, 28), x max: 1.0 , x min: 0.0 , y:[2]
	  x.shape: (1, 1, 28, 28), x max: 1.0 , x min: 0.0 , y:[1]
	  x.shape: (1, 1, 28, 28), x max: 1.0 , x min: 0.0 , y:[0]
	  x.shape: (1, 1, 28, 28), x max: 1.0 , x min: 0.0 , y:[4]
	  ...
	  x.shape: (1, 1, 28, 28), x max: 1.0 , x min: 0.0 , y:[5]
	  x.shape: (1, 1, 28, 28), x max: 0.99607843 , x min: 0.0 , y:[6]


	**AutoReloaderTestMNIST(). Reloading test data**


	load_mnist_0001().INTERNAL SWITCH? YES.
	  x.shape: (1, 1, 28, 28), x max: 1.0 , x min: 0.0 , y:[7]
	  x.shape: (1, 1, 28, 28), x max: 1.0 , x min: 0.0 , y:[2]
	  x.shape: (1, 1, 28, 28), x max: 1.0 , x min: 0.0 , y:[1]
	  x.shape: (1, 1, 28, 28), x max: 1.0 , x min: 0.0 , y:[0]
	  x.shape: (1, 1, 28, 28), x max: 1.0 , x min: 0.0 , y:[4]
	  ...

	"""
	from pipeline.data.load_data import AutoReloaderTestMNIST

	au = AutoReloaderTestMNIST(config_data, shuffle=False)

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


def showcase_mnist_0003(config_data, verbose=0):
	print('showcase_mnist_0003()')
	from pipeline.data.load_data import load_mnist_0001

	config_data['data_from_torch']['mnist']['resize'] = (128,128) # set to None for original size
	config_data['data_from_torch']['mnist']['training_mode'] = True
	dataloader = load_mnist_0001(config_data, batch_size=1, download = False, 
		shuffle=False, verbose=0)

	this_iter = iter(dataloader)
	iter_counter = 0 
	n_iter = len(this_iter)

	
	while iter_counter<n_iter:
		iter_counter+=1

		x,y = this_iter.next()
		print(x.shape)
		x1 = x.clone().detach().cpu().squeeze()

		break

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(x1)
	plt.show()