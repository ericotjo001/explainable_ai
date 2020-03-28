from utils.utils import *

def load_cifar_0001(config_data, batch_size=None, train=True, download=False, 
	shuffle=True, verbose=0):
	print('load_cifar_0001()')

	data_config = config_data['data_from_torch']['cifar']
	conf = config_data['general']
	relative_path = data_config['relative_dir']
	root = os.path.join(config_data['working_dir'],relative_path) 
	
	transformations = []
	if data_config['resize'] is not None:
		transformations.append(transforms.Resize(data_config['resize'], interpolation=2))
	transformations.append(transforms.ToTensor())
	transform = transforms.Compose(transformations)

	create_dir_if_not_exist(root)

	# if dataset has been downloaded, it will not be downloaded again
	cifardata = torchvision.datasets.CIFAR10(root, train=train, transform=transform, target_transform=None, download=True)

	if batch_size is None: batch_size = conf['batch_size']

	data_loader = torch.utils.data.DataLoader(cifardata, batch_size=batch_size, shuffle=shuffle, num_workers=1)
	print("  len(data_loader):%s train=%s"%(str(len(data_loader)),str(train)))
	return data_loader


class AutoReloaderTestCIFAR(object):
	def __init__(self, config_data, shuffle=False):
		super(AutoReloaderTestCIFAR, self).__init__()
		self.config_data = config_data
		self.shuffle = shuffle
		self.test_loader = load_cifar_0001(config_data, batch_size=1, train=False, download=True, 
			shuffle=self.shuffle, verbose=0)
		self.this_iter = iter(self.test_loader)
		self.n_test = len(self.test_loader)
		self.counter = 0
		self.n_batch_test = len(self.test_loader)

	def fetch_next_item(self):
		if self.counter >= self.n_test:
			print('\n\n**AutoReloaderTestCIFAR(). Reloading test data**\n\n')
			# shuffle again 
			self.test_loader = load_cifar_0001(self.config_data, batch_size=1, train=False, download=False, 
				shuffle=self.shuffle, verbose=0)
			self.this_iter = iter(self.test_loader)
			self.counter = 0
		self.counter += 1
		return self.this_iter.next()
