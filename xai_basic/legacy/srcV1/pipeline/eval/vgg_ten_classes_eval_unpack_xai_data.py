from pipeline.training.vgg_ten_classes_root import DEFAULT_CONFIG_DATA
from pipeline.eval.eval_unpack_xai_data import *

def view_gallery(config_data=None, BRANCH_DATA=False):
    if config_data is None: config_data = DEFAULT_CONFIG_DATA
    view_gallery_control(config_data, BRANCH_DATA=BRANCH_DATA)
    view_gallery_control(config_data, BRANCH_DATA=BRANCH_DATA,load_ext='clamp.xai',csv_ext='clamp.csv')

def unpack_and_pointwise_process_xai_data(config_data=None, display_decimal_precision=3, do_print=True,
    BRANCH_DATA=False):
    if config_data is None: config_data = DEFAULT_CONFIG_DATA
    unpack_and_pointwise_process_xai_data_control(config_data, display_decimal_precision=display_decimal_precision, do_print=do_print, BRANCH_DATA=BRANCH_DATA,
        load_ext='xai',csv_ext='csv')
    unpack_and_pointwise_process_xai_data_control(config_data, display_decimal_precision=display_decimal_precision, do_print=do_print, BRANCH_DATA=BRANCH_DATA,
        load_ext='clamp.xai',csv_ext='clamp.csv')

def unpack_and_pointwise_process_xai_data_for_roc(config_data=None,BRANCH_DATA=False):
    if config_data is None: config_data = DEFAULT_CONFIG_DATA

    from pipeline.eval.xai_settings import XAI_SETTING_0001, XAI_SETTING_0002
    unpack_and_pointwise_process_xai_data_for_roc_control(XAI_SETTING_0001 ,config_data, BRANCH_DATA=BRANCH_DATA,
        load_ext='xai',csv_ext='csv')
    unpack_and_pointwise_process_xai_data_for_roc_control(XAI_SETTING_0002 ,config_data, BRANCH_DATA=BRANCH_DATA,
        load_ext='clamp.xai',csv_ext='clamp.csv')