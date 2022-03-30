import numpy as np
from OANN.src.models.BONN import DataIndexer

def evaluate_net(net, folder_to_class_mapping, DATA_DIR, dargs, verbose=100):
    from OANN.examples.donut_example import donut_numpy_loader, get_data_dir
    if dargs['data']=='donut_example':
        results = run_eval_(net, folder_to_class_mapping, DATA_DIR, verbose=verbose)
    elif dargs['data']=='big_donut_example':
        results = run_eval_(net, folder_to_class_mapping, DATA_DIR, verbose=verbose)
    else:
        raise NotImplementedError()
    return results

def run_eval_(net, folder_to_class_mapping, DATA_DIR, verbose=100):
    from OANN.examples.donut_example import donut_numpy_loader, get_data_dir
    dataix = DataIndexer(DATA_DIR=DATA_DIR, 
        folder_to_class_mapping=folder_to_class_mapping, 
        elasticsize=9,
        data_fetcher=donut_numpy_loader, 
        init_new=True,)

    indexset=[]
    n_total, n_correct, false_hit = 0, 0, 0
    results = {
        'incorrect_indices': [],
    }
    while True:
        indexset, INFO = dataix.prepare_elasticset_(indexset)
        x_batch,y0_batch = dataix.fetch_data_by_elastic_set(indexset)

        for i,(x,y0) in enumerate(zip(x_batch, y0_batch)):
            y, act, ACTIVATION_STATUS, interp_info = net.forward(x, ALLOW_INTERPOLATION=True,)
            assert(y is not None)
            y_pred = np.argmax(y)

            IS_CORRECT = y_pred==y0
            if verbose>=100 or not IS_CORRECT:
                print(f'{str(indexset[i]):<10} gt:{y0:>3} pred:{y_pred:>5} [{str(IS_CORRECT):>5}] ACTIVATION_STATUS:{ACTIVATION_STATUS}')

            if ACTIVATION_STATUS=='HIT' and not IS_CORRECT:
                false_hit += 1
            if IS_CORRECT: 
                n_correct+=1
            else:
                results ['incorrect_indices'].append(str(indexset[i]))
                if ACTIVATION_STATUS != 'HIT':
                    print('y:',y)
                    print('some interp_info:')
                    for x,y in interp_info.items():
                        if x=='best': continue
                        print(f'  {x:<4}  alpha:{y["alpha"]} actmax:{y["actmax"]}')
            n_total += 1

        indexset = [] # reset
        if INFO['ALL_DATA_USED_UP']:
            print('ALL DATA USED UP!')
            break

    print(f'n_correct/n_total={n_correct}/{n_total}={n_correct/ n_total }')
    print(f'false_hit/n_total={false_hit}/{n_total}={false_hit/n_total}')
    results['n_total'] = n_total
    results['n_correct'] = n_correct
    results['false_hit'] = false_hit
    return results