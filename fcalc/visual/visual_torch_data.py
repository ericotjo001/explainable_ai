from utils.utils import *

def visual_mnist(config_data):
	print('visual_mnist()')
	import pipeline.data.load_data as lo

	config_data['general']['batch_size'] = 9
	data_loader = lo.load_mnist_0001(config_data)
	
	for i, data in enumerate(data_loader,0):
		x,y = data
		x = x.detach().cpu().numpy()
		y = y.numpy()
		stack = np.concatenate([img for img in x],2)
		print('x.shape:%s,y:%s'%(str(x.shape),str(y)))
		print('stack.shape:%s'%(str(stack.shape)))
		plt.figure()
		plt.imshow(stack.squeeze())
		break
	plt.show()