import argparse
	
class ConsoleModeManager(object):
	"""
	Usage in main.py
		import utils.manage_console as mc
		consoleManager = mc.ConsoleModeManager()
		consoleManager.parse_arguments()
		print(consoleManager.arg_dict)
		
		then in console
		python main.py --mode hello
	"""
	def __init__(self):
		super(ConsoleModeManager, self).__init__()
		self.arg_dict = {}

	def parse_arguments(self, list_of_modes = ['mode']):			
		parser = argparse.ArgumentParser(
			formatter_class=argparse.RawDescriptionHelpFormatter, description=None)

		for submode in list_of_modes:
			parser.add_argument('--'+str(submode))
		
		args, unknown_args = parser.parse_known_args()
		
		for submode in list_of_modes:
			self.arg_dict[submode] = getattr(args, submode)


class ThreeTierConsoleModesManager(ConsoleModeManager):
	def __init__(self):
		super(ThreeTierConsoleModesManager, self).__init__()

		THREE_TIER_MODES = ['mode','mode2','mode3', 'config_dir']
		self.parse_arguments(list_of_modes=THREE_TIER_MODES)

class DynamicConsole(ConsoleModeManager):
	def __init__(self, MODES):
		super(DynamicConsole, self).__init__()
		self.parse_arguments(list_of_modes=MODES)
