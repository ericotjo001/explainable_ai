import os
import numpy as np
import matplotlib.pyplot as plt

from .data import N2LDData, DonutData
from .model import SQANN
from .utils import make_layer_setting, standard_evaluation, \
    manage_dir, pickle_save, pickle_load, create_folder_if_not_exist

def plot_and_save_images(args):
    COLLATED_RESULTS = pickle_load(args['DATA_DIR'])

    def plot_histograms(COLLATED_RESULTS, dir):
        display_order = ['mean_error', 'mean_ex_error', 'mean_frac_error','mean_ex_frac_error','N_INTERPOLATED']
        labels = [r'$\bar{e}$', r'$\bar{e}_X$', r'$\bar{f}$', r'$\bar{f}_X$','$N_{interp}$']
        
        font = {'size': 9}
        plt.rc('font', **font)

        plt.figure(figsize=(14,3))
        for i,display_name in enumerate(display_order):
            this_boxplot = [y[display_name] for stdev,y in COLLATED_RESULTS.items()]
            plt.gcf().add_subplot(1,5,i+1)
            plt.gca().boxplot(this_boxplot, flierprops={'marker':'.', 'markersize':2,})
            plt.gca().set_ylabel(labels[i])
            plt.xticks(range(len(this_boxplot)+1), ['']+[str(stdev) for stdev in COLLATED_RESULTS],)
            plt.gca().set_xlim([0.5,len(this_boxplot)+0.5])
            plt.gca().set_ylim([-0.05,None])
            if i==0:
                plt.gca().set_xlabel('test data spread')

            if display_order[i] in ['mean_frac_error', 'mean_ex_frac_error']:
                max_mean = np.max([np.mean(x) for x in this_boxplot])
                plt.gca().twinx()
                ax = plt.gca().boxplot(this_boxplot, flierprops={'marker':'.', 'markersize':2,})
                plt.gca().set_ylim([0.,2.])
                plt.xticks(range(len(this_boxplot)+1), ['']+[str(stdev) for stdev in COLLATED_RESULTS],)
                plt.gca().set_xlim([0.5,len(this_boxplot)+0.5])
                plt.tick_params(axis='y', labelcolor='r')
                plt.gca().set_ylim([-0.05, max_mean*2])
                for _, line_list in ax.items():
                    for line in line_list:
                        line.set_color('r')
                        line.set_alpha(0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(args['FOLDER_DIR'], 'boxplots.jpeg'))

    plot_histograms(COLLATED_RESULTS,args['DATA_DIR'])


def example_dir(args, name='example1'):
    args = manage_dir(args)
    args['FOLDER_DIR'] = os.path.join(args['CKPT_DIR'], name)
    create_folder_if_not_exist(args['FOLDER_DIR'])
    
    DATA_DIR = os.path.join(args['FOLDER_DIR'], '%s.results'%(str(name)))
    args['DATA_DIR'] = DATA_DIR
    return args


def collect_example1(args):
    print('collect_example1()')

    args = example_dir(args)
    if os.path.exists(args['DATA_DIR']):
        print('data exists, loading it for plotting:', args['DATA_DIR'])
        plot_and_save_images(args)
        exit()

    n_trials = 48
    stds = [0.01, 0.05,0.1, 0.15,0.2]
    n_stds = len(stds)

    N = 128
    MAX_LAYER = 24
    a1s = np.linspace(0.001,1.,51)
    a2s = np.linspace(1.,0.7,101)
    layer_settings = {
        # make_layer_setting(a1, a2,  admission_threshold, activation_threshold, max_n)
        i: make_layer_setting(a1s[i], a2s[i], 0.1, 0.9, ) for i in range(1,1+MAX_LAYER) # BEST NOW
    }

    COLLATED_RESULTS = {}
    for j, stdev in enumerate(stds):
        COLLATED_RESULTS[stdev] = {'N_INTERPOLATED':[], 'mean_error':[], 'mean_frac_error':[],
            'mean_ex_error':[], 'mean_ex_frac_error':[],
        }   
        for i in range(n_trials):
            update_text = '%s/%s %s/%s'%(str(j+1), str(n_stds),str(i+1),str(n_trials))
            print('%-64s'%(update_text), end='\r')

            np.random.seed(i)
            ld = N2LDData(N, vmin=-1.0, vmax=1., sd=0.1, test_sd=stdev, show_fig_and_exit=0)
            X,Y = ld.X, ld.Y
            X_test, Y_test = ld.X_test, ld.Y_test

            net = SQANN(layer_settings, N,)
            net.fit_data(X,Y,verbose=0)
            RESULTS = standard_evaluation(X_test,Y_test,net, verbose=0)

            for x,y in RESULTS.items():
                COLLATED_RESULTS[stdev][x].append(y)

    print()
    pickle_save(COLLATED_RESULTS, args['DATA_DIR'])
    print('saved at DATA_DIR:', args['DATA_DIR'])
    plot_and_save_images(args)
    
def collect_example2(args):
    print('collect_example2()')

    args = example_dir(args, name='example2')
    if os.path.exists(args['DATA_DIR']):
        print('data exists, loading it for plotting:', args['DATA_DIR'])
        plot_and_save_images(args)
        exit()

    n_trials = 48
    stds = [0.01, 0.05,0.1, 0.15,0.2]
    n_stds = len(stds)

    N = 128
    MAX_LAYER = 24
    a1s = np.linspace(0.001,1.,51)
    a2s = np.linspace(1.,0.7,101)
    layer_settings = {
        # make_layer_setting(a1, a2,  admission_threshold, activation_threshold, max_n)
        i: make_layer_setting(a1s[i], a2s[i], 0.1, 0.9, ) for i in range(1,1+MAX_LAYER) # BEST NOW
    }

    COLLATED_RESULTS = {}
    for j, stdev in enumerate(stds):
        COLLATED_RESULTS[stdev] = {'N_INTERPOLATED':[], 'mean_error':[], 'mean_frac_error':[],
            'mean_ex_error':[], 'mean_ex_frac_error':[],
        }   
        for i in range(n_trials):
            update_text = '%s/%s %s/%s'%(str(j+1), str(n_stds),str(i+1),str(n_trials))
            print('%-64s'%(update_text), end='\r')

            np.random.seed(i)
            ld = DonutData(N, test_sd=stdev, show_fig_and_exit=0)
            X,Y = ld.X, ld.Y
            X_test, Y_test = ld.X_test, ld.Y_test

            net = SQANN(layer_settings, N,)
            net.fit_data(X,Y,verbose=0)
            RESULTS = standard_evaluation(X_test,Y_test,net, verbose=0)

            for x,y in RESULTS.items():
                COLLATED_RESULTS[stdev][x].append(y)

    print()
    pickle_save(COLLATED_RESULTS, args['DATA_DIR'])
    print('saved at DATA_DIR:', args['DATA_DIR'])
    plot_and_save_images(args)