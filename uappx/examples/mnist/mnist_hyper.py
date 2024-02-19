from src.utils import parse_bool_from_string, strbool_description, readjust_bools
from .mnist_prep import prep_data_and_dirs, prep_deep_neural_network_and_data_loader, get_admission_th
from .dnn import device
import matplotlib.pyplot as plt
import os, joblib
import numpy as np

def mnist_hyper_(dargs, parser, BOOLS):
    parser.add_argument('--n_per_class', nargs='+', default=120) 
    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)    

    folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs,)
    dnn, mnist_img_loader = prep_deep_neural_network_and_data_loader(dargs, parser, BOOLS, DIRS, device=device)

    from datetime import datetime
    now = datetime.now()
    dated_showcase = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(DIRS['HYPER_FOLDER_DIR'],exist_ok=True)
    HYPER_RESULT_DIR = os.path.join(DIRS['HYPER_FOLDER_DIR'], 'hyper_%s.result'%(str(dated_showcase)))
    HYPER_BOXPLOT_DIR = os.path.join(DIRS['HYPER_FOLDER_DIR'], 'box_%s.png'%(str(dated_showcase)))

    from src.model.kabedonn import KABEDONN    
    settings = {'init_new':True,
            'folder_to_class_mapping': folder_to_class_mapping,
            'DATA_DIR': DIRS['DATA_DIR'],
            'n_class':len(folder_to_class_mapping),
            'kwidth': None, 
            'data_fetcher': mnist_img_loader,
            'interpolator_settings': None,
            'activation_threshold': 0.999,
            'admission_threshold':get_admission_th,
        }
    fittingconfig={'print_final_info':False,'balance_test': False,
        'qconfig': {
            'mode': 'scrambledfirstn',
            'classes': range(10),
            'firstn': [100]*10,
        }
    }
    from src.model.eval import evaluate_on_test_data
    eval_settings = {
        'DIRS':DIRS,
        'folder_to_class_mapping': folder_to_class_mapping,
        'data_fetcher': mnist_img_loader,
    }    

    kwidths = [8, 16, 32, 64]
    labels = [str(k) for k in kwidths]
    labels.insert(0,'')

    accs = {k:[] for k in kwidths}
    for kwidth in kwidths:
        for i in range(8):
            settings['kwidth'] = kwidth 
            print('==== kwidth:%s [%s] ===='%(str(kwidth),str(i)))
            net = KABEDONN(**settings)                    
            qlist = net.fit_data(config=fittingconfig, verbose=0)
            net.evaluate_and_finetune_on_train_data(qlist=qlist, verbose=0)

            TEST_EVAL_RESULT = evaluate_on_test_data(net, **eval_settings)
            TEST_EVAL_RESULT.print_accuracy()
            acc = TEST_EVAL_RESULT.correct/TEST_EVAL_RESULT.total
            # acc = np.random.normal(0,1,)
            accs[kwidth].append(acc)

    joblib.dump(accs, HYPER_RESULT_DIR)

    plt.figure()
    plt.boxplot([y for x,y in accs.items()], 
        flierprops={'marker':'.', 
        'markeredgecolor':'r',
        'markerfacecolor':'r', 
        'markersize':10,}
    )
    plt.xticks( range(len(kwidths)+1), labels)
    plt.xlim([0.5,None])
    plt.gca().set_xlabel('k')

    plt.savefig(HYPER_BOXPLOT_DIR)
    # plt.show()
