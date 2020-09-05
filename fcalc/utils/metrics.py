from utils.utils import *

class ClassAccuracy():
	def __init__(self):
		super(ClassAccuracy, self).__init__()
		self.N = 0
		self.n_correct_pred = 0
		
	def update_acc(self,y1,y0):
		assert(isinstance(y1,int))
		assert(isinstance(y0,int))
		self.N += 1
		if y1==y0:
			self.n_correct_pred+=1

	def compute_acc(self):
		self.acc = float(self.n_correct_pred)/float(self.N)
		return self.acc

	def display_stats(self, tab_level=0, verbose=250,verbose_threshold=250):
		pm.printvm('Acc().display_stats()',
			tab_level=tab_level,verbose=verbose, verbose_threshold=250)
		itemlist = ['N', 'n_correct_pred', 'acc']
		for x in itemlist:
			val = getattr(self, x)
			pm.printvm('%s : %s'%(str(x),str(val)),
				tab_level=tab_level+1,verbose=verbose, verbose_threshold=verbose_threshold)

