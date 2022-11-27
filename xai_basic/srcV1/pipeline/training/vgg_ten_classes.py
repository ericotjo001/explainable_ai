from pipeline.training.vgg_ten_classes_utils import *
import pipeline.data.prepare_10classes_data as data10
from pipeline.workflow_config import DATA_SIZE_VGG

def training_vgg_ten_classes(config_data=None, realtime_update=True):
    if config_data is None: config_data = DEFAULT_CONFIG_DATA
    print('training_vgg_ten_classes(). training scheme:%s'%(str(config_data['training_scheme'])))

    MODEL_DIR, INFO_DIR, CACHE_FOLDER_DIR = prepare_save_dirs(config_data)
    if config_data['training_scheme'] == 'regular_evaluation':
        BRANCH_MODEL_DIR, BRANCH_INFO_DIR = prepare_branch_dirs(MODEL_DIR, config_data) 
        print('  Starting regular evaluation for\n    %s'%(str(BRANCH_MODEL_DIR)))
        
    net, evaluator = initiate_or_load_model(MODEL_DIR, INFO_DIR, config_data)
    optimizer = optimizer_setup(config_data['learning']['adam'], net, mode='adam')
    criterion = nn.CrossEntropyLoss()
    
    # allow longer training and higher tolerance for low validation result 
    evaluator.early_stopping_limit = 48
    evaluator.refresh_fraction = 0.3
    evaluator.reshape_size = DATA_SIZE_VGG

    if config_data['training_scheme'] == 'regular_evaluation':
        evaluator.preloaded_val_dataset = preloading_validation_datasets(CACHE_FOLDER_DIR, config_data)
 
    if config_data['training_data_scheme'] == 'shard_by_shard':
        for epoch in range(config_data['n_epoch']):
            random_shards = list(range(1,1+config_data['training_data']['number_of_data_shards']))
            random.shuffle(random_shards)
            for n_th_shard_iter, k in enumerate(random_shards):
                this_dataset = data10.load_dataset_from_a_shard(k, CACHE_FOLDER_DIR, config_data['data_cache_name'],
                    reshape_size=DATA_SIZE_VGG)
                trainloader = DataLoader(dataset=this_dataset, shuffle=True, batch_size=config_data['batch_size'])        
                n = len(trainloader)
                for i, data in enumerate(trainloader, 0):
                    net.train()
                    update_text = 'epoch: %s/%s. shard: %s/%s (%s/%s) '%(str(epoch+1), str(config_data['n_epoch']), str(n_th_shard_iter+1), 
                        str(config_data['training_data']['number_of_data_shards']), str(i+1), str(n))
                    optimizer.zero_grad()
                    x, y0 = data
                    x = x.to(torch.float).to(this_device)
                    y0 = y0.to(this_device)

                    y = net(x)
                    loss = criterion(y,y0.to(torch.long))
                    loss.backward()            
                    optimizer.step()

                    evaluator.iter += 1
                    # print('evaluator.iter:%s local iter:%s [%s]'%(str(evaluator.iter), str(i),str((evaluator.iter)%config_data['save_every_n_iter'])))
                    if config_data['training_scheme'] == 'continuous':
                        evaluator.compute_running_average_loss(loss)
                    elif config_data['training_scheme'] == 'regular_evaluation':
                        if evaluator.iter%config_data['eval_every_n_iter']==0:
                            evaluator.evaluate_and_save(net, BRANCH_MODEL_DIR, CACHE_FOLDER_DIR, config_data, this_device=this_device)
                        update_text = update_text + 'iter last.update/current: %s/%s best acc.:%s e.stop:%s/%s'%(str(evaluator.last_updated_iter), \
                            str(evaluator.iter),str(round(evaluator.best_acc,3)),str(evaluator.early_stopping_counter),str(evaluator.early_stopping_limit))
                    else:
                        raise RuntimeError('Invalid training scheme.')    

                    if realtime_update: print('  %-96s'%(str(update_text)), end='\r')
                    if config_data['training_scheme'] == 'regular_evaluation':
                        if evaluator.early_stopping: 
                            evaluator.preloaded_val_dataset = None # reset memory (do not save validation dataset)
                            evaluator.pickle_data(evaluator, BRANCH_INFO_DIR, tv=(0,0,VERBOSE_THRESHOLD))
                            return
                            
        if config_data['training_scheme'] == 'continuous':
            evaluator.preloaded_val_dataset = None
            do_save(evaluator, net, MODEL_DIR, INFO_DIR)
        elif config_data['training_scheme'] == 'regular_evaluation':
            evaluator.early_stopping_msg = 'No early stopping signal is triggered.'
            evaluator.evaluate_all_shards(net, config_data)
            evaluator.preloaded_val_dataset = None
            evaluator.pickle_data(evaluator, BRANCH_INFO_DIR, tv=(0,0,VERBOSE_THRESHOLD))

        print('\n-end- Final status update:')                
        print('  %-96s'%(str(update_text)))
    else:
        raise RuntimeError('Invalid training data scheme.')
