DEFAULT_CONFIG_DATA = {
    'folder_dir':'checkpoint/resnet34adj_0001',
    'model_name':'resnet34adj_0001',
    'xai_mode': 'Saliency',
}

from .shared_dependencies import *

def get_stats_across_samples(df, mean_precision=5, std_precision=3,sample_mode=None, exclude_empty=False):
    # For consistency, use this for dataframe df that contains 1 model and 1 xai_method

    if exclude_empty:
        df = df[df['y0']<9] # assuming ten classes with class = 9 being empty

    if sample_mode is None:
        Pmu = df['Pmu'].to_numpy()
        Rmu = df['Rmu'].to_numpy()
        Amu = df['Amu'].to_numpy()
        Pu = df['Pu'].to_numpy()
        Ru = df['Ru'].to_numpy()
        Au = df['Au'].to_numpy()
    elif sample_mode=='only_correct_pred':
        Pmu = df[df['pred_is_correct']==1]['Pmu'].to_numpy()
        Rmu = df[df['pred_is_correct']==1]['Rmu'].to_numpy()
        Amu = df[df['pred_is_correct']==1]['Amu'].to_numpy()
        Pu = df[df['pred_is_correct']==1]['Pu'].to_numpy()
        Ru = df[df['pred_is_correct']==1]['Ru'].to_numpy()
        Au = df[df['pred_is_correct']==1]['Au'].to_numpy()
    else:
        raise Exception('Invalid sample mode.')
    

    sstat = {
        'sAvePmu': round(np.mean(Pmu),mean_precision),
        'sStdPmu': round(np.var(Pmu)**0.5,std_precision),
        'sAveRmu': round(np.mean(Rmu),mean_precision), 
        'sStdRmu': round(np.var(Rmu)**0.5,std_precision),
        'sAveAmu': round(np.mean(Amu),mean_precision), 
        'sStdAmu': round(np.var(Amu)**0.5,std_precision),
        'sAvePu': round(np.mean(Pu),mean_precision),
        'sStdPu': round(np.var(Pu)**0.5,std_precision),
        'sAveRu': round(np.mean(Ru),mean_precision), 
        'sStdRu': round(np.var(Ru)**0.5,std_precision),
        'sAveAu': round(np.mean(Au),mean_precision), 
        'sStdAu': round(np.var(Au)**0.5,std_precision),
    }
    data_columns = {
        'Pmu':Pmu, 'Rmu':Rmu, 'Amu':Amu,
        'Pu': Pu, 'Ru': Ru, 'Au':Au,
    }
    return sstat, data_columns

def gather_basic_results_per_csv(config_data=None, fontsize=8, do_plot=True):
    if config_data is None: config_data = DEFAULT_CONFIG_DATA

    csv_dir = '%s/%s/XAI_results/%s_%s.csv'%(str(config_data['folder_dir']), str(config_data['model_name']),
        str(config_data['model_name']), str(config_data['xai_mode']))
    df = pd.read_csv(csv_dir)

    sstat, data_columns = get_stats_across_samples(df)

    Nrow, Ncol = 1, 3
    font = {'size': fontsize}
    matplotlib.rc('font', **font)   
    fig = plt.figure(figsize=(14,5))

    fig.add_subplot(Nrow,Ncol,1)
    plt.gca().scatter(data_columns['Pmu'], data_columns['Rmu'],3, label='$R_{\mu}$')
    plt.gca().scatter(data_columns['Pmu'], data_columns['Amu'],3, label='$A_{\mu}$')
    plt.gca().set_xlabel('$P_{\mu}$')
    title_prec = '$mean(P_{\mu})$=%s, $std(P_{\mu})$=%s'%(str(sstat['sAvePmu']), str(sstat['sStdPmu']))
    title_recall = '$mean(R_{\mu})$=%s, $std(R_{\mu})$=%s'%(str(sstat['sAveRmu']), str(sstat['sStdRmu']))
    title_acc = '$mean(A_{\mu})$=%s, $std(A_{\mu})$=%s'%(str(sstat['sAveAmu']), str(sstat['sStdAmu']))
    plt.gca().set_title('%s\n%s\n%s'%(title_prec,title_recall, title_acc))
    plt.gca().set_xlim([0.,1.])
    plt.gca().set_ylim([0.,1.])
    plt.legend(title='y axis')
    
    fig.add_subplot(Nrow,Ncol,2)
    plt.gca().scatter(data_columns['Pu'], data_columns['Ru'],3, label='$R_{best}$')
    plt.gca().scatter(data_columns['Pu'], data_columns['Au'],3, label='$A_{best}$')
    plt.gca().set_xlabel('$P_{best}$')
    title_prec = '$mean(P_{best})$=%s, $std(P_{best})$=%s'%(str(sstat['sAvePu']), str(sstat['sStdPu']))
    title_recall = '$mean(R_{best})$=%s, $std(R_{best})$=%s'%(str(sstat['sAveRu']), str(sstat['sStdRu']))
    title_acc = '$mean(A_{best})$=%s, $std(A_{best})$=%s'%(str(sstat['sAveAu']), str(sstat['sStdAu']))
    plt.gca().set_title('%s\n%s\n%s'%(title_prec,title_recall, title_acc))
    plt.gca().set_xlim([0.,1.])
    plt.gca().set_ylim([0.,1.])
    plt.legend(title='y axis')
    
    fig.add_subplot(Nrow,Ncol,3)
    plt.gca().hist(df['Au1'])
    plt.gca().set_title('%s'%('thresholds count'))
    
    if do_plot: plt.show()