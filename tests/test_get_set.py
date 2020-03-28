class GGWP(object):
	def __init__(self):
		super(GGWP, self).__init__()
		self.a = 1
		self.b = 2
	def get_b(self):
		return getattr(self, 'b')
	def set_d(self):
		print('setting d.')
		setattr(self, 'd', 4)	

gg = GGWP()
print('setting c externally.')
setattr(gg, 'c', 3)
y = getattr(gg, 'c')
print('getting c externally. c=',y)
y1 = gg.get_b()
print('getting b by internal method. y1 = gg.b ==',y1)
gg.set_d()
print('gg.d:',gg.d)