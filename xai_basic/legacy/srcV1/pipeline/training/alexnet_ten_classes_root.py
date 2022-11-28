from pipeline.training.shared_dependencies import *
from model.adjusted_alexnet import AdjAlexnet

from pipeline.workflow_config import DEFAULT_CONFIG_DATA
DEFAULT_CONFIG_DATA['model_name'] = 'alexnetadj_0001'
DEFAULT_CONFIG_DATA['batch_size'] = 16
DEFAULT_CONFIG_DATA['learning'] = {
    'adam':{'lr':1e-4, 'weight_decay':1e-5,},
}

from pipeline.data.prepare_10classes_data import DEFAULT_DATA_CONFIG_DATA
for xkey, x in DEFAULT_DATA_CONFIG_DATA.items():
    DEFAULT_CONFIG_DATA[xkey] = x