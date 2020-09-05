import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
from utils.matplotlib_manager import MultiAxesFigure

x = np.array(range(100)).astype(float)

maf = MultiAxesFigure()
maf.set_axis_by_pos(221, (x,x), plot_mode='scatter', title='linear', marker_size=3)
maf.set_axis_by_pos(221, (x,100-x), plot_mode='scatter', marker_size=3,marker_color='r')
maf.set_axis_by_pos(222, (x,x**2), plot_mode='scatter', title='squared', ylim=(0,20000), marker_size=3)
maf.set_axis_by_pos(223, (x,np.exp(-x)), plot_mode='scatter', title='exp', marker_size=3)
maf.set_axis_by_pos(224, (range(15), [1]*15), plot_mode='plot', 
	xlabel='x',ylabel='y', xlim=(None,20), ylim=(0,5))
 
fig = plt.figure()
for pos in maf.data_by_pos:
	data_at_this_pos = maf.data_by_pos[pos]
	setattr(fig,'ax'+str(pos), fig.add_subplot(pos))
	this_ax = getattr(fig,'ax'+str(pos))
	for d in data_at_this_pos:
		maf.plot_data(this_ax, d)
fig.tight_layout()
plt.show()