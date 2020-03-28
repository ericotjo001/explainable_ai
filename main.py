from utils.utils import *
from utils.config_data import config_data
import pipeline.data.entry as da
import pipeline.training.entry as tr
import pipeline.evaluation.entry as ev
import pipeline.lrp.entry as lrp
import pipeline.drivethru.entry as dt
import pipeline.custom_sequence.entry as cus
import pipeline.captum.entry as cap
import pipeline.testing_sequence.entry as teq
import visual.entry as vis


def main():
	config_data['console_mode'] = arg_dict['mode']
	config_data['console_submode'] = arg_dict['submode']
	config_data['console_subsubmode'] = arg_dict['subsubmode']

	if arg_dict['mode'] is None or arg_dict['mode']=='info': 
		print(DESCRIPTION); return
	elif arg_dict['mode'] == 'data':
		da.select_data_mode(config_data)
	elif arg_dict['mode'] == 'visual':
		vis.select_visual_mode(config_data)
	elif arg_dict['mode'] =='training':
		tr.select_training_mode(config_data)
	elif arg_dict['mode'] == 'evaluation':
		ev.select_evaluation_mode(config_data)
	elif arg_dict['mode'] == 'lrp':
		lrp.select_lrp_mode(config_data)
	elif arg_dict['mode'] == 'captum':
		cap.select_captum_mode(config_data)
	elif arg_dict['mode'] == 'drivethru':
		dt.select_drivethru_mode(config_data)
	elif arg_dict['mode'] == 'custom_sequence':
		cus.select_custom_sequence_mode(config_data)
	elif arg_dict['mode'] == 'testing_sequence':
		# this is hidden from usage description
		teq.select_teq_mode(config_data)

	else:
		print('(!!) Invalid mode. Choose one of the following modes.')
		print(DESCRIPTION)


if __name__=='__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
		description=DESCRIPTION)
	parser.add_argument('--mode', help='mode. See utils/utils.py')
	parser.add_argument('--submode', help='submode.')
	parser.add_argument('--subsubmode', help='subsubmode.')
	args = parser.parse_args()
	arg_dict = {
		'mode': args.mode,
		'submode': args.submode,
		'subsubmode': args.subsubmode
	}
	
	main()