from utils.utils import *
import pipeline.custom_sequence.sequence_utils as ut
from utils.logger import TimePrinter
tp = TimePrinter()

from pipeline.training.train_smallnet_mnist import train_smallnet_mnist_0001
def train_smallnet_mnist_0001_timed(run_number, config_data, tab_level=0, verbose=0):
	start = time.time()
	train_smallnet_mnist_0001(config_data, tab_level=tab_level, verbose=verbose)
	end = time.time()
	time_secs = end - start
	tp.print_smh('run_number_'+ str(run_number) ,start, end, if_x_times=(10.,100.),
		verbose=verbose, tab_level=tab_level, verbose_threshold=None)
	ut.print_partition()
	return time_secs

from pipeline.evaluation.eval_smallnet_mnist import eval_smallnet_mnist_0001_overfitting	
def eval_smallnet_mnist_0001_overfitting_timed(config_data, tab_level=0, verbose=0):
	start = time.time()
	eval_smallnet_mnist_0001_overfitting(config_data, tab_level=tab_level, verbose=verbose)
	end = time.time()
	time_secs = end - start
	tp.print_smh('eval_overfitting' ,start, end, if_x_times=(10.,100.),
		verbose=verbose, tab_level=tab_level+1, verbose_threshold=None)
	ut.print_partition()
	return time_secs	



from pipeline.evaluation.eval_smallnet_mnist import eval_smallnet_mnist_0002_test
def eval_smallnet_mnist_0002_test_timed(config_data, tab_level=0, verbose=250):
	start = time.time()
	eval_smallnet_mnist_0002_test(config_data, tab_level=tab_level, verbose=verbose)
	end = time.time()
	time_secs = end - start
	tp.print_smh('eval_test' ,start, end, if_x_times=(10.,100.),
		verbose=verbose, tab_level=tab_level+1, verbose_threshold=None)
	ut.print_partition()
	return time_secs
	