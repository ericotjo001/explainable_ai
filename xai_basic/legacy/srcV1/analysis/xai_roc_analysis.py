from .shared_dependencies import *
import matplotlib

DEFAULT_CONFIG_DATA = {
    'folder_dir':'checkpoint/resnet34adj_0001',
    'model_name':'resnet34adj_0001',
    'xai_modes': ['Saliency'],
    'branch_name_labels': [1],
}

def mass_analysis_0002(config_data=None, do_plot=True,  csv_ext='csv',
    xlim=(0.,1.0), ylim=(0.,1.0),
    markersize=None,fontsize=10,
    figsize=None,legend_bbox_to_anchor=(1.04,1), set_legend=True):
    if config_data is None: config_data = DEFAULT_CONFIG_DATA

    font = {'size'   : fontsize}
    matplotlib.rc('font', **font)
    
    legend_labels = []
    color_scheme = np.linspace(0,1,len(config_data['xai_modes']))
    marker = '+'
    fig = plt.figure(figsize=figsize)
    for i, xai_mode in enumerate(config_data['xai_modes']):
        color=(1-color_scheme[i],0,color_scheme[i])  
        for j, branch_no in enumerate(config_data['branch_name_labels']):
            branch_name = '%s.%s'%(str(config_data['model_name']),str(branch_no))
            csv_dir = '%s/%s/XAI_results/roc_%s_%s.%s'%(str(config_data['folder_dir']), str(branch_name),
                str(branch_name), str(xai_mode), str(csv_ext))
            df = pd.read_csv(csv_dir)
            # print(df)
            X, Y = df['recalls'].to_numpy(), df['FPRs'].to_numpy()
  
            if j==0: 
                thislegend=xai_mode
            else:
                thislegend=None

            plt.gca().plot(X, Y, markersize=markersize, c=color, marker=marker, linewidth=0.3, label=thislegend)


        # alternate_marking
        if marker == '+': marker = '.'
        elif marker == '.': marker = 'x'
        elif marker == 'x': marker = '+'
    plt.gca().plot(np.linspace(0,1,100), np.linspace(0,1,100), c='k', linestyle='--', linewidth=0.3, label=None)

    if set_legend:
        plt.legend( bbox_to_anchor=legend_bbox_to_anchor, borderaxespad=0)
    plt.gca().set_xlabel('FPR')
    plt.gca().set_ylabel('recall')
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    if do_plot: plt.show()