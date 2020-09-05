import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *


def gg(**kwargs):
	print(kwargs)
	kwargs['a'] +=1
	print(kwargs)
	hh(**kwargs)

def hh(**kwargs):
	print("\nhh()")
	print(kwargs)

gg(a=1)