"""
Printing Manager package
Author: Erico Tjoa
Beta version v2.1 
"""
from collections import OrderedDict


class PrintingManager(object):
	def __init__(self):
		super(PrintingManager, self).__init__()

	def findAll(self, input_string, ch):
		return [i for i, letter in enumerate(input_string) if letter == ch]
		
	def print(self, string_input, tab_level=0, tab_shape='  '):
		# tab_level (int)
		if tab_level is None: tab_level = 0
		L = len(tab_shape)
		for i in range(tab_level): 
			if i>=tab_level-1:
				print(tab_shape,end='')
			else:
				print(' '*L,end='')
		print(string_input)

	def printv(self, string_input, tab_level=0, tab_shape='  ',verbose=0, verbose_threshold=None):
		# v: print considering verbosity
		# assume string_input is only oneliner
		if verbose_threshold is None:
			pass
		else:
			assert(isinstance(verbose_threshold, int))
			assert(isinstance(verbose, int))
			if verbose < verbose_threshold:
				return # do not print
		self.print(string_input, tab_level=tab_level, tab_shape=tab_shape)

	def printvm(self, string_input, tab_level=0,verbose=0, verbose_threshold=None, tab_shape='  '):
		"""
		v: print considering verbosity
		m: strings with multiple lines seperated by \n is handled.
		  Does not remove \n if it is in the first letter
		
		example:
			pm = PrintingManager()
			x = 'we\nare\ncats'
			pm.printvm(x,tab_level=2, tab_shape='  ')
		"""
		list_of_nextline_index = self.findAll(string_input,"\n")
		if len(list_of_nextline_index) == 0:
			self.printv(string_input, tab_level=tab_level, tab_shape=tab_shape,\
				verbose=verbose, verbose_threshold=verbose_threshold)
		else:
			temp = [0] + list_of_nextline_index + [len(string_input)]
			current_index = 0
			next_index = temp[1]
			for i in range(1, len(temp)):
				string_with_newline = string_input[current_index:next_index]
				if string_with_newline.find("\n") > -1: string_with_newline = string_with_newline[1:]
				self.printv(string_with_newline,\
					tab_level=tab_level, tab_shape=tab_shape,\
					verbose=verbose, verbose_threshold=verbose_threshold)		
				current_index = next_index
				if i < len(temp)-1:  next_index = temp[i+1]

	def print_list_headtails(self, x, first=3, last=1, do_enumerate=False,
		use_header = None, print_all=False,
		tab_level=0, tab_shape='  ',verbose=0, verbose_threshold=None):
		'''
		first, last (str)
		use_header (str) or None

		import numpy as np
		x = ['baob'+ str(np.random.randint(100)) + '_' + str(i) for i in range(100)]
		pm = PrintingManager()
		pm.print_list_headtails(x, first=3, last=1,
			tab_level=1, tab_shape='  ',verbose=0, verbose_threshold=None)
		'''
		if use_header is not None:
			self.printvm(use_header, tab_level=tab_level,
				verbose=verbose, verbose_threshold=verbose_threshold, tab_shape=tab_shape)
			tab_level += 1

		if print_all:
			n = len(x)
			for i in range(n):
				this_str = '%s'%(str(x[i]))
				self.printvm(this_str, tab_level=tab_level,verbose=verbose, verbose_threshold=verbose_threshold, tab_shape=tab_shape)
			return

		n = len(x)
		for i in range(n):
			if  do_enumerate:
				this_str = '[%s] %s'%(str(i),str(x[i]))
			else:
				this_str = '%s'%(str(x[i]))
			self.print_in_loop(this_str, i, n, first=first, last=last,
				tab_level=tab_level, tab_shape=tab_shape,verbose=verbose, verbose_threshold=verbose_threshold)

	def print_in_loop(self, x, i, n, first=3, last=1,
		tab_level=0, tab_shape='  ',verbose=0, verbose_threshold=None):
		'''
		# Works for loop of list, ANYTHING ELSE?TRY OTHERS
		# does not work for loop of dictionary

		import numpy as np
		x = ['baob'+ str(np.random.randint(100)) + '_' + str(i) for i in range(100)]
		n=len(x)
		pm = PrintingManager()
		for i, x1 in enumerate(x):
			pm.print_in_loop(x1, i, n, first=3, last=5,
				tab_level=0, tab_shape='  ',verbose=0, verbose_threshold=None)
		'''

		if i<first or i>=n-last: 
			self.printvm(x, tab_level=tab_level, tab_shape=tab_shape,
				verbose=verbose, verbose_threshold=verbose_threshold)
		if i==first: 
			self.printvm('...',tab_level=tab_level, tab_shape=tab_shape,
				verbose=verbose, verbose_threshold=verbose_threshold)

	def print_dict(self, this_dict, key_only=True, string_format='%s:  %s', first=3, last=1,
		tab_level=0, tab_shape='  ',verbose=0, verbose_threshold=None):
		'''
		y = {}
		x = ['baob'+ str(np.random.randint(100)) + '_' + str(i) for i in range(100)]
		for x1 in x: y[x1] = str(np.random.randint(1000)) +'_' + x1
		# for xkey in y: print(" %-24s : %s"%(str(xkey),str(y[xkey])))
		pm.print_dict(y, key_only=False, string_format='%s:  %s', first=3, last=1,
			tab_level=0, tab_shape='  ',verbose=0, verbose_threshold=None)
		'''
		n = len(this_dict)
		i = 0
		for xkey in this_dict:
			if key_only: 
				x = str(xkey)
			else:
				x = string_format%(str(xkey),str(this_dict[xkey]))
			if i<first or i>=n-last: 
				self.printvm(x, tab_level=tab_level, tab_shape=tab_shape,
					verbose=verbose, verbose_threshold=verbose_threshold)
			if i==first: 
				self.printvm('...',tab_level=tab_level, tab_shape=tab_shape,
					verbose=verbose, verbose_threshold=verbose_threshold)
			i+=1	

	def print_terminal_dict(self, x,
		tab_level=0, tab_shape='  ',verbose=0, verbose_threshold=None):
		for ykey, y in x.items():
			self.printvm('%s:%s'%(str(ykey),str(y)),tab_level=tab_level, tab_shape=tab_shape,
				verbose=verbose, verbose_threshold=verbose_threshold) 	

	def print_recursive_dict(self,x,
		tab_level=0, tab_shape='  ',verbose=0, verbose_threshold=None):
		# check example in test/test_print_dict_recursive.py
		for ykey, y in x.items():
			if type(y) == type({}) or isinstance(y, OrderedDict):
				self.printvm('%s:'%(str(ykey)),tab_level=tab_level, tab_shape=tab_shape,
					verbose=verbose, verbose_threshold=verbose_threshold) 	
				self.print_recursive_dict(y,
					tab_level=tab_level+1, tab_shape=tab_shape,verbose=verbose, verbose_threshold=verbose_threshold)
			else:
				self.printvm('%s:%s'%(str(ykey),str(y)),tab_level=tab_level, tab_shape=tab_shape,
					verbose=verbose, verbose_threshold=verbose_threshold) 	

	def print_2Dmatrix_format(self, item_list, header=None, cell_string_format='%5s', column_size=5, separator=None,
		tab_level=0, tab_shape='  ', verbose=0, verbose_threshold=None):
		"""
		Example:
		x = np.random.randint(100, size=(24))
		print(x)
		pm.print_2Dmatrix_format(x, cell_string_format='%5s', column_size=None, separator=',',
			tab_level=2, tab_shape='  ', verbose=0, verbose_threshold=None)
		pm.print_2Dmatrix_format(x, cell_string_format='%5s', column_size=5, separator=',',
			tab_level=1, tab_shape='  ', verbose=0, verbose_threshold=None)
		pm.print_2Dmatrix_format(x, cell_string_format='%3s', column_size=7, separator='|',
			tab_level=1, tab_shape='  ', verbose=0, verbose_threshold=None)
		"""	
		if header is not None:
			self.printvm(header, 
				tab_level=tab_level, tab_shape=tab_shape,verbose=verbose, verbose_threshold=verbose_threshold)
	
		if separator is None: separator = ' '
		n = len(item_list)
		if column_size is None: column_size = n

		column_now, current_string = 0, ''
		for i in range(n):
			if column_now < column_size:
				current_string += cell_string_format%(item_list[i])
				if column_now < column_size - 1 and not(i+1==n):
					current_string += separator
				column_now += 1
			if column_now >= column_size or i+1==n:
				self.printv(current_string, 
					tab_level=tab_level, tab_shape=tab_shape,verbose=verbose, verbose_threshold=verbose_threshold)
				column_now, current_string = 0, ''
	

class ShortPrint(PrintingManager):
	"""PrintingManger with abbreviated notation"""
	def __init__(self):
		super(ShortPrint, self).__init__()

	def prints(self, string_input, tv=(0,0,None), tab_shape='  '):
		"""
		tv=(tab_level,verbose, verbose_threshold)
		"""
		self.printvm(string_input, tab_level=tv[0],verbose=tv[1], verbose_threshold=tv[2], tab_shape=tab_shape)
		
	def printd(self, dictionary, tv=(0,0,None), tab_shape='  ', use_header=None):
		tab_level = tv[0]
		if use_header is not None:
			self.prints(use_header,tv=tv)
			tab_level += 1
		self.print_recursive_dict(dictionary,
			tab_level=tab_level, tab_shape=tab_shape,verbose=tv[1], verbose_threshold=tv[2])

