from utils.utils import *

class Logger():
	"""
	Usage, run your program after running the following
	sys.stdout = Logger(full_path_log_file="hello.txt")
	"""
	def __init__(self, full_path_log_file="logfile.log"):
		self.terminal = sys.stdout
		self.log = open(full_path_log_file, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self): pass   


from utils.printing_manager import PrintingManager
class TimePrinter(PrintingManager):
	def __init__(self):
		super(TimePrinter, self).__init__()
	
	def print_smh(self, series_name ,start, end, if_x_times=(10.,100.),
		round_sec=2, round_min=1, round_hr=1,
		verbose=0, tab_level=0, verbose_threshold=None):
		# start, end are both time.time()
		
		self.printvm('series_name: %s'%(series_name),
			verbose=verbose, tab_level=tab_level, verbose_threshold=verbose_threshold)
		
		time_secs = end-start
		
		self.print_one_smh('',time_secs,
			round_sec=round_sec, round_min=round_min, round_hr=round_hr,
			verbose=verbose, tab_level=tab_level+1, verbose_threshold=verbose_threshold)
	
		for i, x in enumerate(if_x_times):
			header = '%8s ==> '%(str(x))
			self.print_one_smh(header ,time_secs*x,
				round_sec=round_sec, round_min=round_min, round_hr=round_hr,
				verbose=verbose, tab_level=tab_level+1, verbose_threshold=verbose_threshold)


	def print_one_smh(self, header, time_secs, 
		round_sec=2, round_min=1, round_hr=1,
		verbose=0, tab_level=0, verbose_threshold=None):
		
		self.printvm('%s%s [s] = %s [min] = %s [hr]'%(
			str(header),
			str(round(time_secs,round_sec)),
			str(round(time_secs/60.,round_min)), 
			str(round(time_secs/3600.,round_hr))
			),
			verbose=verbose, tab_level=tab_level, verbose_threshold=verbose_threshold)