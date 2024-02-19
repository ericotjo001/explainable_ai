import os, joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.utils import parse_bool_from_string, strbool_description, readjust_bools
from .cifar_prep import prep_data_and_dirs, prep_deep_neural_network_and_data_loader, get_admission_th

from .dnn import device

# let's import some functions we defined before from mnist example
from ..mnist.mnist_showcase import pick_relevant_indices, fetch_image_by_class_and_index, get_similar_images

def ces(dargs, parser, BOOLS):
    print('ces: compare, euclidean, selected.')
    print('compare with euclidean dist, selected samples...')
    # see fig. 4 of Understanding Black-box Predictions via Influence Functions, Pang Wei Koh

    parser.add_argument('--redir_id', default=0, type=int, help=None)
    parser.add_argument('--classes', nargs='+', default=[4,9]) 
    parser.add_argument('--firstn', nargs='+', default=[120,120]) 
    parser.add_argument('--assess', nargs='+', default=None, help='y0 idx y0 idx ...') 

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)    

    modelname = 'cifar_%s_%s.pth'%(str('-'.join(dargs['classes'])),str('-'.join(dargs['firstn'])))
    folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs, modelname=modelname)
    dnn, cifar_img_loader = prep_deep_neural_network_and_data_loader(dargs, parser, BOOLS, DIRS, device=device)

    from src.model.kabedonn import KABEDONN
    settings = {'init_new':True,
        'folder_to_class_mapping': folder_to_class_mapping,
        'DATA_DIR': DIRS['DATA_DIR'],
        'n_class':len(folder_to_class_mapping),
        'kwidth':dargs['kwidth'], # for donut data, this is about 3 samples per class in a layer
        'data_fetcher': cifar_img_loader,
        'interpolator_settings': None,
        'activation_threshold': 0.999,
        'admission_threshold':get_admission_th,
    }
    net = KABEDONN(**settings)
    fitting_config = {   
        'print_final_info':True,
        'balance_test': False,    
        'qconfig': {
            'mode': 'firstn',
            'classes': dargs['classes'],
            'firstn': dargs['firstn'],
        }
    }    

    if dargs['assess'] is None:
        # from .cifar_debugtests import redirect_for_debugtests
        # redirect_for_debugtests(dargs, net=net, fitting_config=fitting_config, setting=settings)

        net.fit_data(config=fitting_config)
        print('saving to %s...'%(str(DIRS['MODEL_DIR'])))
        net.ix.data_fetcher = None # so that it is now pickleable
        joblib.dump(net, DIRS['MODEL_DIR'])
    else:
        net = joblib.load(DIRS['MODEL_DIR'])
        net.ix.data_fetcher = cifar_img_loader
        print(net.print_final_())

        assert(len(dargs['assess'])%2==0)
        indices = np.array(dargs['assess'], dtype=int).reshape(-1,2)
        assessing_selected_samples(net, indices, DIRS, modelname)



def assessing_selected_samples(net, indices, DIRS, modelname):
    """
    indices:
     [(y0_1,idx_1), (y0_2,idx_2),...]
    """
    font = {'size': 6}
    plt.rc('font', **font)
    plt.figure()
    
    from datetime import datetime
    now = datetime.now()
    dated_showcase = now.strftime("%Y-%m-%d_%H-%M-%S")

    CES_DIR = os.path.join(DIRS['CKPT_DIR'],'showcase_selected_%s'%(str(modelname)))  
    os.makedirs(CES_DIR, exist_ok=True)
    REPORT_DIR = os.path.join(CES_DIR,dated_showcase+'.txt')

    def offticks():
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

    with open(REPORT_DIR, 'w') as f:
        counter = 0
        print('assessing_selected_samples...')
        elasticset = [(y0,idx) for y0,idx in indices]
        x_batch, y0_batch = net.ix.fetch_data_by_elastic_set(elasticset)
        n_rows = len(y0_batch)
        for i,(x,y0) in enumerate(zip(x_batch,y0_batch)):
            y, OUTPUT_INFO = net.forward(x)
            NODE_INFO = OUTPUT_INFO['NODE_INFO']
            activated_node = OUTPUT_INFO['activated_node']

            y0_sample,idx_sample = indices[i]
            f.write('\n==== index: (%s,%s) pred:%s [%s] ====\n'%(y0_sample,idx_sample, y,str(y==y0_sample)))
            status = net.get_status_by_interpolation(y, OUTPUT_INFO, print_status=False)
            f.write('  data_loc:%s\n'%(str(status['data_loc'])))
            f.write('  parent_node_loc:%s\n'%(str(status['parent_node_loc'])))

            for outi, outc in OUTPUT_INFO.items():
                if outi in ['NODE_INFO','activated_node','act']: continue
                f.write('  %s:%s\n'%(str(outi),str(outc)))
            status = net.get_status_by_interpolation(y, OUTPUT_INFO, print_status=False)

            f.write('NODE INFO:\n'%())
            for outi, outc in NODE_INFO.items():
                f.write('  %s:%s\n'%(str(outi),str(outc)))                

            WR_NODES = activated_node.wr_nodes_
            WR_FILES = [net.ix.get_folder_and_filename_by_index(y0,idx) for y0,idx in WR_NODES]
            SUB_NODES = [x for x in activated_node.sub_nodes_]
            SUB_FILES = [net.ix.get_folder_and_filename_by_index(y0,idx) for y0,idx in SUB_NODES]

            f.write('wr_nodes_:\n')
            if len(WR_NODES)>0:
                for a,b in zip(WR_NODES, WR_FILES):
                    f.write('  %-10s:%s\n'%(str(a),str(b)))
            else:
                f.write('  <empty>\n')

            f.write('subnode:\n')
            if len(SUB_NODES)>0:
                for a,b in zip(SUB_NODES, SUB_FILES):
                    f.write('  %-10s:%s\n'%(str(a),str(b)))
            else:
                f.write('  <empty>\n')


            img = fetch_image_by_class_and_index(y0_sample, idx_sample, net)
            INDICES = pick_relevant_indices(activated_node, OUTPUT_INFO, n_pick=5)
            y0_inf,idx_inf = INDICES['influential']
            img_inf = fetch_image_by_class_and_index(y0_inf, idx_inf,net)
            similar_imgs = get_similar_images(INDICES, net)

            plt.gcf().add_subplot(n_rows,3,3*counter+1)
            plt.gca().imshow(img, )
            plt.gca().set_ylabel('%s'%(str(elasticset[i])))
            offticks()
            
            plt.gcf().add_subplot(n_rows,3,3*counter+2)
            plt.gca().imshow(img_inf,)
            plt.gca().set_ylabel('sub:%s'%(str(int(NODE_INFO['subactivation']))))
            offticks()

            plt.gcf().add_subplot(n_rows,3,3*counter+3)
            if len(similar_imgs)>0:
                plt.gca().imshow(similar_imgs,)
            offticks()
            counter+=1

            all_signals = net.forward_get_all_signals(x)
    
    dir1 = os.path.join(CES_DIR,dated_showcase+'.png')
    plt.savefig(dir1)
    plt.close()
    print('saving %s...'%(str(dir1)))


    plt.figure(figsize=(5,12))
    counter = 0
    memo = {}
    for i,(x,y0) in enumerate(zip(x_batch,y0_batch)):
        main_signals, side_signals = net.forward_get_all_signals(x)

        norms, acts = [], []
        elasticset, activations = [], []
        norms2, acts2 = [], []
        elasticset2, activations2 = [], []
        for layer_, signal in main_signals.items():
            elasticset.extend(signal['indices'])
            activations.extend(signal['act'])

        for (this_class,this_index), act in zip(elasticset,activations):
            if (this_class,this_index) not in memo:
                x_batch, y0_batch = net.ix.fetch_data_by_elastic_set([(this_class,this_index)])
                memo[(this_class,this_index)] = x_batch[0], y0_batch[0]
            this_x, this_y0 = memo[(this_class,this_index)]

            norm = np.linalg.norm(this_x-x)
            norms.append(norm)
            acts.append(act)

        if side_signals is not None:
            if side_signals['subactivation']:
                elasticset2.extend(side_signals['indices'])
                activations2.extend(side_signals['act'])

        for (this_class,this_index), act in zip(elasticset2,activations2):
            if (this_class,this_index) not in memo:
                x_batch, y0_batch = net.ix.fetch_data_by_elastic_set([(this_class,this_index)])
                memo[(this_class,this_index)] = x_batch[0], y0_batch[0]
            this_x, this_y0 = memo[(this_class,this_index)]

            norm = np.linalg.norm(this_x-x)
            norms2.append(norm)
            acts2.append(act)
        
        plt.gcf().add_subplot(n_rows,1,counter+1)
        plt.gca().scatter(norms, acts, 3)
        plt.gca().scatter(norms2, acts2, 3)

        plt.gca().set_xlim([-1,None])
        plt.gca().set_ylim([0,1.1])
        plt.gca().set_yticks([0,1.])
        plt.gca().set_ylabel('%s'%(str(elasticset[i])))
        counter+=1
    plt.tight_layout()
    dir2 = os.path.join(CES_DIR,dated_showcase+'_profile.png')
    plt.savefig(dir2)
    plt.close()
    print('saving to %s...'%(str(dir2)))