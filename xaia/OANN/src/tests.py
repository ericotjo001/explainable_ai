import numpy as np

def redirect_for_testing_indexmanagement(dargs, TOGGLES, **kwargs):
    """
    This testing will show you how data are integrated layer by layer.
    At each layer, before the size of elastic size reach the intended size,
      more data will be collected and tested.
    Repeat the process until all data are used up.
    """
    print('redirect_for_testing_indexmanagement...')
    ix = kwargs['net'].ix
    elasticset = []
    print('data sizes')
    ix.print_(mode='datasizes')
    print('init layer status')
    ix.print_(mode='datalayerstatus')

    def process_dummy_data(elasticset, p_fail=0.1):
        new_elastic_set, decomissioned = [], []
        for i, (classlabel, current_index) in enumerate(elasticset):
            coin = np.random.uniform(0,1)

            if coin>p_fail:
                new_elastic_set.append((classlabel, current_index))
            else:
                decomissioned.append((classlabel, current_index))
        return new_elastic_set, decomissioned

    p_fail = 0.1
    current_layer = 1
    for i in range(18):
        elasticset, INFO = ix.prepare_elasticset_(elasticset)
        print(f'\niter:{i}:\n{elasticset}')
        elasticset, decomissioned = process_dummy_data(elasticset, p_fail=p_fail)
        print(f'{elasticset} \\   {decomissioned}')
        ix.update_decommisioned_data(decomissioned)

        if len(elasticset)>=ix.elasticsize:
            print(f'MAKING LAYER {current_layer}')
            ix.integrate_data_indices(elasticset, layer_k=current_layer)
            current_layer+=1

            # reset
            elasticset,p_fail = [], 0.1

        if INFO['ALL_DATA_USED_UP']:
            print('ALL DATA USED UP!')
            ix.integrate_data_indices(elasticset, layer_k=current_layer)
            break            
        print('layer status:')
        ix.print_(mode='datalayerstatus')
        print('datapointer:')
        ix.print_(mode='datapointer')
        p_fail -= 0.02

def redirect_for_testing_fetch_data_by_elasticset(dargs, TOGGLES, **kwargs):
    print('redirect_for_testing_fetch_data_by_elasticset...')
    elasticset = []
    net = kwargs['net']

    current_layer = 1
    while True:
        elasticset, INFO = net.ix.prepare_elasticset_(elasticset)
        x_batch,y0_batch = net.ix.fetch_data_by_elastic_set(elasticset)
        
        print(elasticset)
        print('  x_batch.shape:',np.array(x_batch).shape)
        print('  y0_batch:',y0_batch)
        
        elasticset = []
        if INFO['ALL_DATA_USED_UP']:
            print('ALL DATA USED UP!')
            break

def redirect_for_testing_data_fitting(dargs, TOGGLES, **kwargs):
    print('redirect_for_testing_data_fitting...')

    net = kwargs['net']
    net.fit_data(max_iter=None, verbose=100)


GOODNEWS = """==== GOOD NEWS ====
IF WE GET HERE, it means all data that still MISS ACTIVATIONS are data that have 
been decomissioned (not vice versa).
"""
def redirect_for_prediction(dargs, TOGGLES, **kwargs):
    print('redirect_for_prediction...')
    print('interpolation?', kwargs['ALLOW_INTERPOLATION'])

    from OANN.examples.donut_example import donut_numpy_loader, get_data_dir
    from OANN.src.models.BONN import DataIndexer
    
    indexset = []
    trainix = DataIndexer(DATA_DIR=get_data_dir(), 
        folder_to_class_mapping={f'class{i}':i for i in [0,1,2]}, 
        elasticsize=9,
        data_fetcher=donut_numpy_loader, 
        init_new=True,)

    verbose = 20
    net = kwargs['net']
    FITTING_INFO = net.fit_data(max_iter=None, verbose=verbose)

    net.print_fitting_info('layer_settings', *[], verbose=verbose)

    ALL_WRONG_IS_MISS = []
    MISSED_INDICES = []
    print('\nstart testing...')
    n_total, n_correct = 0, 0
    while True:
        indexset, INFO = trainix.prepare_elasticset_(indexset)
        x_batch,y0_batch = trainix.fetch_data_by_elastic_set(indexset)

        for i,(x,y0) in enumerate(zip(x_batch, y0_batch)):
            y, act, ACTIVATION_STATUS, interp_info = net.forward(x, ALLOW_INTERPOLATION=kwargs['ALLOW_INTERPOLATION'],)
            
            y = np.argmax(y) if y is not None else 'none'
            is_correct = y==y0
            print(f'{str(indexset[i]):>10} gt:{y0:>3} pred:{y:>5} [{str(is_correct):>5}], ACTIVATION_STATUS:{ACTIVATION_STATUS}')

            if (not is_correct) and ACTIVATION_STATUS=='HIT':
                raise RuntimeError('DANGEROUS!')  
            if is_correct:
                n_correct += 1
            else:
                ALL_WRONG_IS_MISS.append(ACTIVATION_STATUS=='MISS')

            if ACTIVATION_STATUS=='MISS':
                MISSED_INDICES.append(indexset[i])

            n_total+=1

        indexset = [] # reset
        if INFO['ALL_DATA_USED_UP']:
            print('ALL DATA USED UP!')
            break

    if kwargs['ALLOW_INTERPOLATION']:
        print(f'n_correct/n_total={n_correct}/{n_total}={n_correct/ n_total }')
    else:
        print('ALL_WRONG_IS_MISS:', np.all(ALL_WRONG_IS_MISS))
        print('MISSED_INDICES:\n', MISSED_INDICES)
        print('decomissioned:\n', FITTING_INFO['decomissioned'])
        print('unresolvable:\n', FITTING_INFO['unresolvable'])
        assert(len(FITTING_INFO['unresolvable'])==0)
        assert(set(MISSED_INDICES).issubset(FITTING_INFO['decomissioned'] ))

        print(GOODNEWS)