from utils.utils import *

class PercentileMeanPower(object):
	def __init__(self):
		super(PercentileMeanPower, self).__init__()
		
	def sorted_sign_split(self, x, mp_threshold=0.8, verbose=0, tab_level=0):
		# x numpy array of size (C, any spatial dimension), C: channel size
		# e.g. for 2D image (C,H,W)
		# Assume positive and negative mean powers are the same if any one of them is not defined (i.e. the list of values are empty)
		temp = x.reshape(-1)

		temp = temp[~np.isnan(temp)]
		
		x_positive = np.sort(temp[temp>=0])
		x_negative = np.sort(-temp[temp<0])

		if len(x_positive) > 0:
			positive_mean_power = self.compute_mean_power(x_positive,threshold=mp_threshold,verbose=0, tab_level=0)	
			if len(x_negative) > 0:
				negative_mean_power = self.compute_mean_power(x_negative,threshold=mp_threshold,verbose=0, tab_level=0)	
			else:
				negative_mean_power = positive_mean_power
		else:
			if len(x_negative) > 0:
				negative_mean_power = self.compute_mean_power(x_negative,threshold=mp_threshold,verbose=0, tab_level=0)
				positive_mean_power = negative_mean_power
			else:
				raise Exception('utils/metrics_for_lrp_output.py. sorted_sign_split() finds empty positive and negative lists')

		pm.printvm('positive_mean_power=%s, negative_mean_power=%s'%(str(positive_mean_power), str(negative_mean_power)),
			verbose=verbose, verbose_threshold=250, tab_level=tab_level)
		return positive_mean_power, negative_mean_power

	def compute_mean_power(self, x, threshold=0.7, verbose=0, tab_level=0):
		# x is 1D numpy array with non-negative entries
		n = len(x)
		x = np.sort(x)
		n_th = int(np.ceil(n*threshold))
		if n_th==1:
			return 1.0
		if n_th>=n: 
			n_th = n-1

		if len(x[n_th:]) == 0 or len(x[:n_th]) == 0:
			pm.printvm('compute_mean_power()', tab_level=tab_level)
			pm.printvm('len(x):%s\nn_th:%s'%(str(n),str(n_th)), tab_level=tab_level+1)
			pm.printvm('x.shape:%s'%(str(x.shape)), tab_level=tab_level+1)
			raise Exception('compute_mean_power(). DIVIDE BY ZERO ERROR UPCOMING')

		upper_strength = np.mean(x[n_th:])
		lower_strength = np.mean(x[:n_th])

		if upper_strength==0:
			return 0.
		if lower_strength==0:
			lower_strength = 1e-9 # buffer

		meanpower = upper_strength/lower_strength # no need for normalization
		pm.printvm('n_th:%s\n%s/%s =%s'%(str(n_th), str(upper_strength), 
			str(lower_strength), str(meanpower)),
			verbose=verbose, verbose_threshold=250, tab_level=tab_level)
		return meanpower

	def compute_mean_power_example(self, sample_size=100, array_size=100, threshold=0.8):
		pm.printvm('threshold=%s'%(str(threshold)),tab_level=0)

		X_dict = {}
		pos_list = [231,232,233,234,235,236]
		n_plot = len(pos_list)
		for i in range(n_plot): X_dict[i] = [] 
		title_list = [
			'uniform int dist [0,100]',
			'uniform int dist squared',
			'uniform int dist\ntop 0.9 uniform amplified',
			'uniform int dist\ntop 0.9 random amplified',
			'uniform int scaled\n[No effect]',
			'uniform int translated +20\n'
		]
		assert(n_plot==len(title_list))


		for i in range(sample_size):
			x = np.random.randint(0,100,size=(array_size)).astype(float)
			X_dict[0].append(self.compute_mean_power(x,threshold=threshold))
			
			x2 = x**2
			X_dict[1].append(self.compute_mean_power(x2,threshold=threshold))
			
			x3 = x.copy()
			x3[int(array_size*0.9):] *= 2
			X_dict[2].append(self.compute_mean_power(x3,threshold=threshold))
			
			x4 = x.copy()
			x4[int(array_size*0.9):] *= (np.abs(np.random.normal(0.,2.)) + 1)
			X_dict[3].append(self.compute_mean_power(x4,threshold=threshold))

			x5 = 10*x 
			X_dict[4].append(self.compute_mean_power(x5,threshold=threshold))

			x6 = x + 20
			X_dict[5].append(self.compute_mean_power(x6,threshold=threshold))


		fig = plt.figure()
		for i, pos, title in zip(range(n_plot),pos_list,title_list):
			setattr(fig,'ax'+str(pos),fig.add_subplot(pos))
			this_ax = getattr(fig,'ax'+str(pos))
			this_ax.scatter(range(len(X_dict[i])),X_dict[i],3)
			this_ax.plot(range(len(X_dict[i])),[1.0]*len(X_dict[i]),c='r')
			this_ax.set_ylim(0,10)
			this_ax.set_title(title)
		fig.tight_layout()

