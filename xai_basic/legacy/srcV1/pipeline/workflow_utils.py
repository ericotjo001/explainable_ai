import pipeline.data.prepare_10classes_data as data10
import os

def get_data(config_data, CACHE_FOLDER_DIR, realtime_update):
    # Training data
    data10.create_or_load_data_shards(CACHE_FOLDER_DIR, config_data['validation_data_cache_name'], 
        n_shard=config_data['val_data']['number_of_data_shards'] , 
        n_per_shard= config_data['val_data']['number_of_data_per_shard'], realtime_update=realtime_update)
    
    # Validation data
    data10.create_or_load_data_shards(CACHE_FOLDER_DIR, config_data['data_cache_name'], 
        n_shard=config_data['training_data']['number_of_data_shards'] , 
        n_per_shard= config_data['training_data']['number_of_data_per_shard'], realtime_update=realtime_update)
    
    # Evaluation data
    data10.create_or_load_data_shards(CACHE_FOLDER_DIR, config_data['test_data_cache_name'], 
        n_shard=config_data['test_data']['number_of_data_chunks'] , 
        n_per_shard= config_data['test_data']['number_of_data_per_chunk'], 
        realtime_update=realtime_update, include_xai_variables=True)

def check_model_for_xai_exist(branch_number, MODEL_DIR, config_data):
    BRANCH_FOLDER = MODEL_DIR[:MODEL_DIR.find('.model')] + '.%s'%(str(branch_number))
    r_eval_model_dir =  os.path.join(BRANCH_FOLDER,'%s.%s.model'%(str(config_data['model_name']),str(config_data['branch_name_label'])) )

    if os.path.exists(r_eval_model_dir + '.optim'): xai_model_dir = r_eval_model_dir + '.optim'
    elif os.path.exists(r_eval_model_dir + '.noptim'): xai_model_dir = r_eval_model_dir + '.noptim'
    else: raise RuntimeError(msg)

    return xai_model_dir