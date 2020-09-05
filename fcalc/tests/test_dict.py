import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

save = {}

for i in range(10):
	if i not in save:
		save[i] = {}
	save[i] = 'a'+str(i)

print(save)