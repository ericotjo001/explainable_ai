from ..utils import *

def save_acc_table(data, model, AX_CLASS_ACC_DIR, VIS_DIR):
    with open(AX_CLASS_ACC_DIR) as f:
        acc_data = json.load(f)

    df = {}
    columns = ['Baseline', 'Saliency', 'InputXGradient', 'LayerGradCam', 'Deconvolution','GuidedBackprop', 'DeepLift']
    label = f'{data}_{model}'
    for ax_method_ in columns:
        if ax_method_ == 'Baseline':
            ax_method = 'None'
        else:
            ax_method = ax_method_

        acc = acc_data[f'{label}_{ax_method}']['acc']
        df[ax_method_] = [acc]

    df = pd.DataFrame(df)
    df.index = [label]
    df.to_csv(os.path.join(VIS_DIR, 'acc.csv'))

def save_vis_dir(PROJECT_DIR, VIS_DIR, DEBUG_TOGGLE_VIS_ITER=False):
    scores_dirs = glob.glob(f"{PROJECT_DIR}/*.scores")
    arrange_results = {}
    arrange_results_softmax = {}
    for scores_dir in scores_dirs:
        co_scores_info = joblib.load(scores_dir)
        
        # print(scores_dir)
        # print(co_scores_info['ax_method'])
        # print(co_scores_info['co_scores'][:4])

        ax_method = co_scores_info['ax_method']
        if not ax_method in arrange_results:
            arrange_results[ax_method] = {True: [], False:[]}
        if not ax_method in arrange_results_softmax:
            arrange_results_softmax[ax_method] = {True: [], False:[]}

        for i,(co_score, co_score_softmax, isCorrect) in enumerate(co_scores_info['co_scores']): 
            arrange_results[ax_method][isCorrect].append(co_score)
            arrange_results_softmax[ax_method][isCorrect].append(co_score_softmax)

            if DEBUG_TOGGLE_VIS_ITER:
                if i>=10: break

    ax_method_list = ['Saliency', 'InputXGradient', 'LayerGradCam', 'Deconvolution','GuidedBackprop', 'DeepLift']
    n_methods = len(ax_method_list)    

    score_list, score_softmax_list = [],[]
    TAG = []
    for ax_method in ax_method_list:
        for isCorrect in [True, False]:
            print(f"{ax_method}_{isCorrect}")
            print('  no of nan:', np.sum(np.isnan(arrange_results[ax_method][isCorrect])))
            print('  no of nan (softmax):', np.sum(np.isnan(arrange_results_softmax[ax_method][isCorrect])))
 
            score_list.append([a for a in arrange_results[ax_method][isCorrect] if ~np.isnan(a)])
            score_softmax_list.append([a for a in arrange_results_softmax[ax_method][isCorrect] if ~np.isnan(a)])

    boxplotlabels = ['',]
    for ax_method in ax_method_list:
        boxplotlabels = boxplotlabels + [ax_method+'.1', ax_method+'.0']
    font = {'size': 9}
    plt.rc('font', **font)
    plt.figure(figsize=(10,4))
    plt.gcf().add_subplot(1,2,1)
    plt.gca().boxplot(score_list, flierprops={'marker':'.', 
        'markeredgecolor':'r','markerfacecolor': (1,0,0,0.2), 'markersize':3,})
    plt.gca().set_xticks( range(2*n_methods+1), boxplotlabels, rotation=60)
    plt.gca().set_ylabel('co score')
    plt.xlim([0.5, None])

    plt.gcf().add_subplot(1,2,2)
    plt.gca().boxplot(score_softmax_list , flierprops={'marker':'.', 
        'markeredgecolor':'r','markerfacecolor': (1,0,0,0.2), 'markersize':3,})
    plt.gca().set_xticks( range(2*n_methods+1), boxplotlabels, rotation=60)
    plt.gca().set_ylabel('co score')
    plt.gca().set_title('with softmax')
    plt.xlim([0.5, None])

    plt.tight_layout()
    boxplot_dir = os.path.join(VIS_DIR, 'boxplot.png') 
    plt.savefig(boxplot_dir)
    plt.close()

    print(f"results saved to {boxplot_dir}")

    font = {'size': 6}
    plt.rc('font', **font)
    plt.figure(figsize=(12,3))
    for k, ax_method in enumerate(ax_method_list):
        plt.gcf().add_subplot(2,6,1+k)
        plt.gca().hist(arrange_results[ax_method][True], alpha=0.2, label='correct')
        plt.gca().hist(arrange_results[ax_method][False], alpha=0.2, label='wrong')
        plt.gca().set_title(f'{ax_method}')
        plt.gca().set_xlabel('CO score')

        plt.gcf().add_subplot(2,6,7+k)
        plt.gca().hist(arrange_results_softmax[ax_method][True], alpha=0.2, label='correct')
        plt.gca().hist(arrange_results_softmax[ax_method][False], alpha=0.2, label='wrong')
        plt.gca().set_title(f'{ax_method} - with softmax')
        plt.gca().set_xlabel('CO score')

    plt.tight_layout()
    plt.legend()
    hist_dir = os.path.join(VIS_DIR, f'co-hist.png')
    plt.savefig(hist_dir)
    plt.close()       

    print(f"results saved to {hist_dir}")