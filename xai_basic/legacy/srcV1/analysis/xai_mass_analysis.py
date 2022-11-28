DEFAULT_CONFIG_DATA = {
    'folder_dir':'checkpoint/resnet34adj_0001',
    'model_name':'resnet34adj_0001',
    'xai_modes': ['Saliency'],
    'branch_name_labels': [1],
}

from .shared_dependencies import *
from analysis.xai_analysis import get_stats_across_samples
import matplotlib

def mass_analysis_0001(config_data=None, do_plot=True, xlim=(0.,1.0), ylim=(0.,1.0), csv_ext='csv', 
    sample_mode=None, exclude_empty=False,
    plot_mode='recall_vs_precision', 
    markersize=3, figsize=None, legend_bbox_to_anchor=(1.04,1), fontsize=10,
    print_values=True, set_legend=True):
    """
    sample averages of recall vs precision
    """
    if config_data is None: config_data = DEFAULT_CONFIG_DATA

    font = {'size'   : fontsize}
    matplotlib.rc('font', **font)
    
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(111)

    color_scheme = np.linspace(0,1,len(config_data['xai_modes']))
    legend_labels = []
    if plot_mode == 'recall_vs_precision':
        Xkey, Ykey = 'sAvePmu', 'sAveRmu'
        Xerrkey, Yerrkey = 'sStdPmu', 'sStdRmu'
        xlabel, ylabel = '$P_{\mu}$', '$R_{\mu}$'
    elif plot_mode == 'acc_vs_precision':
        Xkey, Ykey = 'sAvePmu', 'sAveAmu'
        Xerrkey, Yerrkey = 'sStdPmu', 'sStdAmu' 
        xlabel, ylabel = '$P_{\mu}$', '$A_{\mu}$'    
    elif plot_mode == 'recall_vs_precision_best':
        Xkey, Ykey = 'sAvePu', 'sAveRu'
        Xerrkey, Yerrkey = 'sStdPu', 'sStdRu'
        xlabel, ylabel = '$P_{u}$', '$R_{u}$'
    elif plot_mode == 'acc_vs_precision_best':
        Xkey, Ykey = 'sAvePu', 'sAveAu'
        Xerrkey, Yerrkey = 'sStdPu', 'sStdAu' 
        xlabel, ylabel = '$P_{u}$', '$A_{u}$'        
    else:
        raise RuntimeError('Invalid plot_mode.')

    marker = '+'
    for i, xai_mode in enumerate(config_data['xai_modes']):
        X, Y = [], []
        Xerr, Yerr = [], []

        if print_values: print('%-16s '%(xai_mode), end= '')
        for j, branch_no in enumerate(config_data['branch_name_labels']):
            branch_name = '%s.%s'%(str(config_data['model_name']),str(branch_no))
            csv_dir = '%s/%s/XAI_results/%s_%s.%s'%(str(config_data['folder_dir']), str(branch_name),
                str(branch_name), str(xai_mode), str(csv_ext))
            df = pd.read_csv(csv_dir)
            sstat, data_columns = get_stats_across_samples(df, sample_mode=sample_mode, exclude_empty=exclude_empty)

            color=(1-color_scheme[i],0,color_scheme[i])            
            # plt.gca().scatter(sstat['sAvePmu'], sstat['sAveRmu'], 3 , c=[color])
            
            X.append(sstat[Xkey])
            Y.append(sstat[Ykey])
            Xerr.append(sstat[Xerrkey])
            Yerr.append(sstat[Yerrkey])

            if print_values: print('(%7s,%7s) '%(str(sstat[Xkey]),str(sstat[Ykey])), end='')
        if print_values: print()

        plt.gca().errorbar(X, Y, xerr=Xerr, yerr=Yerr, c=color, elinewidth=0.2, fmt=marker,markersize=markersize)
        plt.gca().set_xlim(xlim)
        plt.gca().set_ylim(ylim)
        legend_labels.append(xai_mode)

        # alternate_marking
        if marker == '+': marker = '.'
        elif marker == '.': marker = 'x'
        elif marker == 'x': marker = '+'
    if set_legend:
        plt.legend(legend_labels, bbox_to_anchor=legend_bbox_to_anchor, borderaxespad=0)
    plt.gca().set_xlabel(xlabel)        
    plt.gca().set_ylabel(ylabel)
    if do_plot: plt.show()

