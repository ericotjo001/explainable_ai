import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

class GG(object):
	"""docstring for GG"""
	def __init__(self, arg):
		super(GG, self).__init__()
		self.arg = arg


gg = GG('abc')
gg.a = 'a'
print(gg.arg)
print(gg.a)