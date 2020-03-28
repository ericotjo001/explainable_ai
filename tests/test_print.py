import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *


x = np.random.randint(100, size=(24))
print(x)
pm.print_2Dmatrix_format(x, cell_string_format='%5s', column_size=None, separator=',',
	tab_level=2, tab_shape='  ', verbose=0, verbose_threshold=None)
pm.print_2Dmatrix_format(x, cell_string_format='%5s', column_size=5, separator=',',
	tab_level=1, tab_shape='  ', verbose=0, verbose_threshold=None)
pm.print_2Dmatrix_format(x, cell_string_format='%3s', column_size=7, separator='|',
	tab_level=1, tab_shape='  ', verbose=0, verbose_threshold=None)