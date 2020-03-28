import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
import utils.matplotlib_manager as mm

ibi = mm.ImshowByIter()

def main():
	path_to_save_folder = 'multiClassimg'
	filename_prefix = 'test'

	iter_values = np.arange(0,11,1)
	dict_by_iter = create_dict_by_iter(iter_values, data_type='numpy', array_size=(28,28))
	ibi.plot_by_iter(dict_by_iter, path_to_save_folder, filename_prefix, 
		ext='.jpg', column_size=5)

def create_dict_by_iter(iter_values,data_type='numpy', array_size=(28,28), mu=0, sd=1):
	this_dict = {}
	for this_iter in iter_values:
		if data_type=='numpy':
			this_dict[this_iter] = this_iter+ np.random.normal(mu, sd,size=array_size)
		elif data_type=='float':
			this_dict[this_iter] = this_iter + np.random.normal(mu,sd)
	return this_dict

if __name__=='__main__':
	main()