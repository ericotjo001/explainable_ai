import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *

def main():

	from matplotlib import offsetbox
	from sklearn import (manifold, datasets, decomposition, ensemble,
	                     discriminant_analysis, random_projection, neighbors)
	digits = datasets.load_digits(n_class=6)
	X = digits.data
	y = digits.target
	n_samples, n_features = X.shape
	n_neighbors = 30

	print('X.shape:%s'%(str(X.shape)))
	print('y.shape:%s'%(str(y.shape)))
	# print(X[0])
	# print(digits.images[0])
	# return

	# ----------------------------------------------------------------------
	# Scale and visualize the embedding vectors
	def plot_embedding(X, title=None):
	    x_min, x_max = np.min(X, 0), np.max(X, 0)
	    X = (X - x_min) / (x_max - x_min)

	    plt.figure()
	    ax = plt.subplot(111)
	    for i in range(X.shape[0]):
	        plt.text(X[i, 0], X[i, 1], str(y[i]),
	                 color=plt.cm.Set1(y[i] / 10.),
	                 fontdict={'weight': 'bold', 'size': 9})

	    # if hasattr(offsetbox, 'AnnotationBbox'):
	    #     # only print thumbnails with matplotlib > 1.0
	    #     shown_images = np.array([[1., 1.]])  # just something big
	    #     for i in range(X.shape[0]):
	    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
	    #         if np.min(dist) < 4e-3:
	    #             # don't show points that are too close
	    #             continue
	    #         shown_images = np.r_[shown_images, [X[i]]]
	    #         imagebox = offsetbox.AnnotationBbox(
	    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
	    #             X[i])
	    #         ax.add_artist(imagebox)
	    plt.xticks([]), plt.yticks([])
	    if title is not None:
	        plt.title(title)


	# t-SNE embedding of the digits dataset
	print("Computing t-SNE embedding")
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	t0 = time.time()
	X_tsne = tsne.fit_transform(X)

	plot_embedding(X_tsne,
	               "t-SNE embedding of the digits (time %.2fs)" %
	               (time.time() - t0))
	plt.show()


if __name__ == '__main__':
	main()

