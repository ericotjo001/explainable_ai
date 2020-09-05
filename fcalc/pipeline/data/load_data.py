from utils.utils import *

def load_mnist_0001(config_data, batch_size=None, download = False, shuffle=True, verbose=0):
	print('load_mnist_0001()..')
	"""
	load_mnist_0001().INTERNAL SWITCH? YES.
	  x.shape:(4, 1, 28, 28), x max:1.0, x min:0.0, y:[2 9 1 6]
	  x.shape:(4, 1, 28, 28), x max:1.0, x min:0.0, y:[8 0 4 4]
	  x.shape:(4, 1, 28, 28), x max:1.0, x min:0.0, y:[9 2 0 0]
	  x.shape:(4, 1, 28, 28), x max:1.0, x min:0.0, y:[4 8 2 8]
	  x.shape:(4, 1, 28, 28), x max:1.0, x min:0.0, y:[2 2 6 0]
	  ...
	"""	

	conf = config_data['general']
	mnist_config = config_data['data_from_torch']['mnist']

	transformations = []
	if mnist_config['resize'] is not None:
		transformations.append(transforms.Resize(mnist_config['resize'], interpolation=2))
	transformations.append(transforms.ToTensor())
	transform = transforms.Compose(transformations)

	###################################
	relative_path = mnist_config['relative_dir']
	training_mode = mnist_config['training_mode']
	root = os.path.join(config_data['working_dir'],relative_path) 
	create_dir_if_not_exist(root)

	if not download:
		if not os.path.exists(os.path.join(root,'MNIST')): 
			print('MNIST data NOT FOUND. Proceed with download now...\n')
			download = True

	mnist_data = torchvision.datasets.MNIST(root, train=training_mode, transform=transform, target_transform=None, download=download)

	if batch_size is None: 
		batch_size = conf['batch_size']
		print('  batch_size is None: hence batch_size = config_data[general][batch_size]')
		print('    batch_size:%s'%(str(batch_size)))

	data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=shuffle, num_workers=1)

	for i, data in enumerate(data_loader,0):
		if verbose<250: break
		x,y = data

		x1 = x.clone().detach().cpu().numpy()
		pm.print_in_loop('x.shape: %s, x max: %s , x min: %s , y:%s'%(
			str(x1.shape),
			str(np.max(x1)),
			str(np.min(x1)),
			str(y.clone().detach().cpu().numpy())), 
			i, len(data_loader), first=5, last=2,
		tab_level=1, verbose=verbose, verbose_threshold=250)
		if i>=10: break

	return data_loader

class AutoReloaderTestMNIST(object):
	"""
	Class for autoreloading test data one by one
	"""
	def __init__(self, config_data, shuffle=False):
		super(AutoReloaderTestMNIST, self).__init__()

		self.config_data = config_data
		self.config_data['data_from_torch']['mnist']['training_mode'] = False
		
		self.shuffle = shuffle
		self.test_loader = load_mnist_0001(config_data, shuffle=self.shuffle, batch_size=1, verbose=0)
		self.this_iter = iter(self.test_loader)
		self.n_test = len(self.test_loader)
		self.counter = 0
		self.n_batch_test = len(self.test_loader)

	def fetch_next_item(self):
		if self.counter >= self.n_test:
			print('\n\n**AutoReloaderTestMNIST(). Reloading test data**\n\n')
			# shuffle again 
			self.test_loader = load_mnist_0001(self.config_data, shuffle=self.shuffle, batch_size=1, verbose=0)
			self.this_iter = iter(self.test_loader)
			self.counter = 0
		self.counter += 1
		return self.this_iter.next()