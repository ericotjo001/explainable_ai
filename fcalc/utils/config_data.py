from utils.debug_switches import *

config_data = {
	'working_dir': 'D:/Desktop@D/meim2venv/fcalc' ,# "D:/Desktop@D/meim2venv/fcalc",
	'general':{
		'batch_size': 4,
		'epoch':1
	},
	'data_from_torch':{
		'mnist':{
			'relative_dir': 'data/mnist', 
			'training_mode': True, # Bool. True: train, False: test,
			'resize':  None # (140,140) # None #
		},
		'cifar':{
			'relative_dir': "data/cifar",
			'resize': None # (256,256) # None #
		}
	},
	'training':{
		'series_name': '000001'
	},
	'lrp':{
		'n_th_run_to_load': 3,
		'relprop_mode': 'relprop1'  
	},
	'visual':{
		'n_th_run_to_load': 3,
		'series_name': '000001' 
	},
	'drivethru':{
		'no_of_evals_per_run':4, # NO_OF_EVALUATION_DESIRED
		'eval_every_n_iter': 7,
		'n_of_test_data_per_eval':100,
		'n_of_test_data_per_LRP_eval':24,
		'lrp_on_training_data':False # rather than test data
	},
	'learning':{
		'mechanism':'adam',
		'SGD':{
			'learning_rate':0.001,
			'momentum': 0.9,
			"weight_decay":1e-05
		},
		'adam':{
			'learning_rate':0.0001,
			'betas': [0.5,0.9]
		}
	}
}

if NSCC_MODE:
	# for job sent to NSCC, Singapore.
	config_data['working_dir'] = '/mnt'
	config_data['training']['series_name'] = 'N' + config_data['training']['series_name']

"""
lrp:
  relprop_mode:
  	relprop1
  	relprop2

'data_from_torch':{
	'mnist':{
		'resize': 
			None
			(140,140)
	'cifar':{
		'resize': 
			None
			(256,256)
"""