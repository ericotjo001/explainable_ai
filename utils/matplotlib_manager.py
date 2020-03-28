"""
Matplotlib Manager package
Author: Erico Tjoa
Beta version v0.1
"""

from utils.printing_manager import PrintingManager
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import os
pm = PrintingManager()

class MultiAxesFigure(Figure):
	def __init__(self):
		super(MultiAxesFigure, self).__init__()
		self.data_by_pos = {}

	def set_axis_by_pos(self, pos, data, plot_mode='scatter', **kwargs):
		# pos e.g. 221
		if pos not in self.data_by_pos:
			self.data_by_pos[pos] = []
		self.data_by_pos[pos].append({
			'data':data,
			'plot_mode':plot_mode,
			'title':kwargs.get('title'),
			'marker_size': kwargs.get('marker_size'),
			'marker_color': kwargs.get('marker_color'),
			'xlabel': kwargs.get('xlabel'),
			'ylabel': kwargs.get('ylabel'),
			'xlim': kwargs.get('xlim'),
			'ylim': kwargs.get('ylim')
		})

	def plot_data(self,this_ax, data_at_this_pos):
		if data_at_this_pos['plot_mode'] == 'scatter':
			x, y = data_at_this_pos['data']
			this_ax.scatter(x,y,
				s=data_at_this_pos['marker_size'], 
				c=data_at_this_pos['marker_color'] )
		elif data_at_this_pos['plot_mode'] == 'plot':
			x, y = data_at_this_pos['data']
			this_ax.plot(x,y, c=data_at_this_pos['marker_color'] )
		else:
			raise Exception('Invalid plot_mode')
			
		this_ax.set_xlabel(data_at_this_pos['xlabel'])
		this_ax.set_ylabel(data_at_this_pos['ylabel'])
		this_ax.set_xlim(data_at_this_pos['xlim'])
		this_ax.set_ylim(data_at_this_pos['ylim'])
		this_ax.set_title(data_at_this_pos['title'])

class ImshowByIter(object):
	def __init__(self):
		super(ImshowByIter, self).__init__()

	def plot_by_iter(self, dict_by_iter, path_to_save_folder, filename_prefix, 
		ext='.jpg', column_size=5):
		n = len(dict_by_iter)
		img_matrix = []
		for i, (this_iter, this_img) in enumerate(dict_by_iter.items()): 
			img_matrix.append(this_img)
			if (i+1)%column_size==0 or i+1==n:
				img_matrix = np.concatenate(img_matrix,axis=1)
				print(img_matrix.shape)
				# fig = plt.figure()

				# ax = fig.add_subplot(211)
				# ax.imshow()
				# ax2 = fig.add_subplot(212)

				# filename = os.path.join(path_to_save_folder, filename_prefix+str('_')+str(this_iter)+ext)
				# plt.saveifg(filename)
				# plt.close()
				img_matrix = []