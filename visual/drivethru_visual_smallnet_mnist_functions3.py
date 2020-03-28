from utils.utils import *

def visualize_filter_effect_over_iter(saliency_by_groundtruth_vs_iter,path_to_save_folder,
	verbose=0, tab_level=0):
	pm.printvm('visualize_filter_effect_over_iter()', tab_level=tab_level)
	print_saliency_by_groundtruth_vs_iter_data(saliency_by_groundtruth_vs_iter, verbose=0, tab_level=tab_level)

	N_ROW_PER_IMG = 7
	VIEWING_FACTOR = 100.

	for this_iter, this_dict in saliency_by_groundtruth_vs_iter.items():
		for y0, list_of_lrp_filtered_saliency_package in this_dict.items():
			folder_by_y0 = os.path.join(path_to_save_folder, str(y0))
			if not os.path.exists(folder_by_y0):
				os.mkdir(folder_by_y0)
			
			# for i, lrp_filtered_saliency_package in enumerate(list_of_lrp_filtered_saliency_package):
			n = len(list_of_lrp_filtered_saliency_package)
			n_partition = int(np.ceil(n/N_ROW_PER_IMG))

			for k_th_partition in range(n_partition):
				list_subset_of_lrp_filtered_saliency_package = None
				if k_th_partition+1<n_partition:
					list_subset_of_lrp_filtered_saliency_package = list_of_lrp_filtered_saliency_package[k_th_partition*N_ROW_PER_IMG:(k_th_partition+1)*N_ROW_PER_IMG]
				else:
					list_subset_of_lrp_filtered_saliency_package = list_of_lrp_filtered_saliency_package[k_th_partition*N_ROW_PER_IMG:]

				# one_img, ordered_names, one_col, one_col_pred = prepare_data_for_one_image(k_th_partition, list_subset_of_lrp_filtered_saliency_package)
				# save_images_by_parts(folder_by_y0 , this_iter, y0, k_th_partition, one_img, ordered_names, suffix='', isimg=False)
				one_img_normalized, ordered_names, one_col, one_col_pred, one_col_rep = prepare_data_for_one_image(k_th_partition, list_subset_of_lrp_filtered_saliency_package, normalize=True)
				# save_images_by_parts(folder_by_y0 , this_iter, y0, k_th_partition, one_col, one_col_pred, suffix='_x', isimg=True)
				save_images_by_parts(folder_by_y0 , this_iter, y0, k_th_partition, one_img_normalized, one_col, one_col_pred, one_col_rep, ordered_names, suffix='_norm', isimg=False, VIEWING_FACTOR = VIEWING_FACTOR )	

			
def save_images_by_parts(folder_by_y0, this_iter, y0, k_th_partition, one_img, one_col, one_col_pred, one_col_rep, ordered_names, suffix='', isimg=True, VIEWING_FACTOR = 1.):
	
	number = 10000000 + this_iter
	filename = str(y0) + '_iter' + str(number)[1:]  +  '_part_' +str(k_th_partition)
	filename = os.path.join(folder_by_y0, filename)
	# print(filename)

	one_img, cmap, this_title = img_subprocessing(one_img, ordered_names, isimg=False)
	one_col_rep, _, _ = img_subprocessing(one_col_rep, ordered_names, isimg=True)
	one_img_x, cmap_x, this_title_x = img_subprocessing(one_col, [int(x) for x in one_col_pred], isimg=True)

	fig = plt.figure()

	absv = np.abs(np.max(np.abs(one_img.reshape(-1))))
	ax = plt.subplot(121)
	im1 = ax.imshow(one_img, cmap=cmap)
	im1.set_clim(vmin=-absv/VIEWING_FACTOR, vmax=absv/VIEWING_FACTOR)

	if len(one_col_rep.shape) == 3:
		ax.imshow(np.dot(one_col_rep, [0.299, 0.587, 0.114]), alpha=0.5, cmap='gray')
	elif len(one_col_rep.shape) == 2:
		ax.imshow((one_col_rep), alpha=0.5, cmap='gray')
	else:
		print('save_images_by_parts(). SKIPPING!')
		return
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_title(this_title)
	set_size(9,9)
	plt.axis('off')
	plt.tight_layout()

	absv_x = np.abs(np.max(np.abs(one_col.reshape(-1))))
	ax2 = plt.subplot(122)
	im2 = ax2.imshow(one_img_x, cmap=cmap_x)
	im2.set_clim(vmin=-absv_x, vmax=absv_x)
	ax2.set_xticklabels([])
	ax2.set_yticklabels([])
	ax2.set_title(this_title_x)
	set_size(9,5)

	plt.axis('off')
	plt.tight_layout()
	# plt.colorbar(im1,ax=ax)
	plt.savefig(filename + suffix +'_vf'+str(VIEWING_FACTOR) + '.jpg')
	plt.close()

def img_subprocessing(one_img, ordered_names, isimg=True):
	if isimg:
		# is_img 
		cmap = None
		one_img = normalize_numpy_array(one_img,target_min=1e-6,target_max=1.)
		one_img = one_img.transpose(1,2,0)
		s = one_img.shape
		if s[2] == 1:
			cmap = 'bwr'
			one_img = one_img[:,:,0]

	else:
		cmap = 'bwr'
		for c, img_one_channel in enumerate(one_img):
			if c==0:
				temp_img = one_img[0]*0
				temp_img += img_one_channel
			else:		
				temp_img += img_one_channel
		one_img = temp_img /len(one_img)

	this_title = '%s'%(str(ordered_names),)
	return one_img, cmap, this_title

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def prepare_data_for_one_image(k_th_partition, list_subset_of_lrp_filtered_saliency_package, normalize=False):
	one_img= []
	ordered_names = []
	one_col, one_col_pred, one_col_rep = [], [], []
	for i, lrp_filtered_saliency_package in enumerate(list_subset_of_lrp_filtered_saliency_package):
		one_row= []
		for filter_type, filtered_img in lrp_filtered_saliency_package.items():
			if filter_type == 'main':
				x,y0,y1 = filtered_img
				# print(x.shape,y0,y1)
				one_col.append(x) 

				one_row_x = []
				for j in range(len(lrp_filtered_saliency_package)-1):
					one_row_x.append(x)
				one_row_x = np.concatenate(one_row_x, axis=2)
				one_col_rep.append(one_row_x)
				one_col_pred.append(y0==y1)
			else: 
				if normalize:
					one_row.append(filtered_img/np.abs(np.max(np.abs(filtered_img.reshape(-1)))))
				else:
					one_row.append(filtered_img)
				if i==0:
					ordered_names.append(filter_type)
		one_row = np.concatenate(one_row, axis=2)
		one_img.append(one_row)
	one_img = np.concatenate(one_img, axis=1)
	one_col = np.concatenate(one_col, axis=1)
	one_col_rep = np.concatenate(one_col_rep, axis=1)
	# print(one_col_pred)
	# print(one_col_rep.shape)
	return one_img, ordered_names, one_col, one_col_pred, one_col_rep


def print_saliency_by_groundtruth_vs_iter_data(saliency_by_groundtruth_vs_iter, verbose=0, tab_level=0):
	pm.printvm('print_saliency_by_groundtruth_vs_iter_data()', tab_level=tab_level)
	
	temp_iter = None
	temp_y0 = None
	for this_iter, this_dict in saliency_by_groundtruth_vs_iter.items():
		pm.printvm(' %8s | %s '%(str(this_iter),str(type(this_dict))), tab_level=tab_level+1, verbose=verbose, verbose_threshold=250)
		if temp_iter is None: temp_iter = this_iter
		for y0, list_of_lrp_filtered_saliency_package in this_dict.items():
			pm.printvm(' %8s | %4s | %s x lrp_filtered_saliency_package '%(str(''),str(y0), str(len(list_of_lrp_filtered_saliency_package))), tab_level=tab_level+1, verbose=verbose, verbose_threshold=250)
			if temp_y0 is None :temp_y0 = y0
		break
	list_of_lrp_filtered_saliency_package = saliency_by_groundtruth_vs_iter[temp_iter][temp_y0]
	a_lrp_filtered_saliency_package = list_of_lrp_filtered_saliency_package[0]

	pm.printvm('a_lrp_filtered_saliency_package:', tab_level=tab_level+1, verbose=verbose, verbose_threshold=250)
	for xkey, this_R in a_lrp_filtered_saliency_package.items():
		if xkey == 'main':
			pm.printvm(' %-20s | %s '%(str(xkey),str(type(this_R))), tab_level=tab_level+2, verbose=verbose, verbose_threshold=250)	
		else:
			pm.printvm(' %-20s | %s '%(str(xkey),str(this_R.shape)), tab_level=tab_level+2, verbose=verbose, verbose_threshold=250)
		"""
		a_lrp_filtered_saliency_package:
		   R                    | (1, 28, 28)
		   clamp_(0.0,0.9)      | (1, 28, 28)
		   clamp_(0.0,0.6)      | (1, 28, 28)
		   clamp_(0.0,0.3)      | (1, 28, 28)
		   amp_1.2              | (1, 28, 28)
		   amp_1.5              | (1, 28, 28)
		   amp_2.0              | (1, 28, 28)
		"""