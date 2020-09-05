from utils.debug_switches import *
from utils.packages import *
from utils.printing_manager import PrintingManager
from utils.messages import *

pm = PrintingManager()

DESCRIPTION="""WELCOME!

For the results in the publication, 
"Generalization on the Enhancement of Layerwise Relevance Interpretability of Deep Neural Network"
run the following:
python main.py --mode custom_sequence --submode smallnet_mnist --subsubmode RXXXX1
python main.py --mode custom_sequence --submode alexnet_mnist --subsubmode RXXXX1

Available modes:

(1) Information
python main.py
python main.py --mode info

(2) Data
python main.py --mode data

(3) Visual
python main.py --mode visual

(4) Training
python main.py --mode training

(5) Evaluation
python main.py --mode evaluation

(6) LRP. Layerwise Relevance Propagation for interpretability.
python main.py --mode lrp

(7) Captum. Pytorch interpretability API (_NOT_IMPLEMENTED_)
python main.py --mode captum

(8) Drive through. This mode performs evaluation and other processing along training.
python main.py --mode drivethru

(X) custom_sequence
python main.py --mode custom_sequence

(X) testing_sequence
python main.py --mode testing_sequence

"""

this_device = torch.cuda.current_device()

def create_dir_if_not_exist(this_dir):
	if not os.path.exists(this_dir):
		os.mkdir(this_dir)

def normalize_numpy_array(x,target_min=-1,target_max=1, 
	source_min=None, source_max=None, verbose = 250):
	'''
	If target_min or target_max is set to None, then no normalization is performed
	'''
	if source_min is None: source_min = np.min(x)
	if source_max is None: source_max = np.max(x)
	if target_min is None or target_max is None: return x
	if source_min==source_max:
		if verbose> 249 : print("normalize_numpy_array: constant array, return unmodified input")
		return x
	midx0=0.5*(source_min+source_max)
	midx=0.5*(target_min+target_max)
	y=x-midx0
	y=y*(target_max - target_min )/( source_max - source_min)
	return y+midx
