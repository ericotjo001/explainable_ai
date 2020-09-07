import os
import pandas as pd
import numpy as np
from pipeline.eval.evaluation_xai_utils import EvaluationPackage
import matplotlib.pyplot as plt


def tabularize_data_point(metrics_collection):
    pointwise_table = {
        'pixel_acc':[],
        'recall': [],
        'precision': [],
        'FPR':[],
        'thresholds': [] ,
        'thresholds0': [] ,
    }
    for metric in metrics_collection:
        for xkey, xitem in metric.items():
            if xkey == 'thresholds': xitem = np.round(xitem,5)
            pointwise_table[xkey].append(xitem)
    pointwise_table = pd.DataFrame(pointwise_table)
    return pointwise_table

def view_gallery_control(config_data, BRANCH_DATA=False,
    load_ext='xai',csv_ext='csv'):    
    
    VERBOSE = 0
    
    xai_mode = config_data['xai_mode']
    model_name = config_data['model_name']
    if not BRANCH_DATA:
        XAI_FOLDER = os.path.join('checkpoint', model_name, 'XAI_results')
        XAI_SAVEDIR = os.path.join(XAI_FOLDER, '%s_%s.%s'%(str(model_name), str(xai_mode), str(load_ext)))
        CSV_SAVEDIR = os.path.join(XAI_FOLDER, '%s_%s.%s'%(str(model_name), str(xai_mode), str(csv_ext)))
        GALLERY_DIR = os.path.join(XAI_FOLDER, 'gallery_%s.%s.%s'%(str(model_name), str(xai_mode), str(load_ext)))
    else:
        XAI_FOLDER = os.path.join('checkpoint', model_name, '%s.%s'%(str(model_name),str(config_data['branch_name_label'])), 'XAI_results')
        XAI_SAVEDIR = os.path.join(XAI_FOLDER, '%s.%s_%s.%s'%(str(model_name), str(config_data['branch_name_label']), str(xai_mode), str(load_ext)))        
        CSV_SAVEDIR = os.path.join(XAI_FOLDER, '%s.%s_%s.%s'%(str(model_name), str(config_data['branch_name_label']), str(xai_mode), str(csv_ext)))
        GALLERY_DIR = os.path.join(XAI_FOLDER, 'gallery_%s.%s_%s.%s'%(str(model_name), 
            str(config_data['branch_name_label']), str(xai_mode), str(load_ext)))
    print('  view_gallery() %s'%(str(XAI_SAVEDIR)))

    if not os.path.exists(GALLERY_DIR): os.mkdir(GALLERY_DIR)

    assert(os.path.exists(CSV_SAVEDIR)) # otherwise, do unpack_and_pointwise_process_xai_data() first
    
    evalpack = EvaluationPackage()
    evalpack = evalpack.load_pickled_data(XAI_SAVEDIR, tv=(1,VERBOSE,250))

    for gallery_label, gallery_items in evalpack.GALLERY.items():
        COLORWISE_HEATMAPS_DIR = os.path.join(GALLERY_DIR, '%s.jpg'%(str(gallery_label)))
        BANDED_HEATMAPS_DIR = os.path.join(GALLERY_DIR, '%s_bands.jpg'%(str(gallery_label)))
        # print(gallery_label, len(gallery_items), ITEM_DIR)
        save_fig_by_gallery_label(gallery_label, gallery_items, COLORWISE_HEATMAPS_DIR)
        save_fig_by_gallery_label_banded_heatmaps(gallery_label, gallery_items, BANDED_HEATMAPS_DIR)


def save_fig_by_gallery_label_banded_heatmaps(gallery_label, gallery_items, ITEM_DIR):
    from pipeline.eval.eval_metrics import XAITranformation3c1c, FiveBandXAIMetric
    xt = XAITranformation3c1c()
    xai = FiveBandXAIMetric()

    fig = plt.figure(figsize=(22,22))
    plt.tight_layout()
    N_COL, N_ROW = 7, 7
    for i, gallery_item in enumerate(gallery_items):
        x = gallery_item['x'] # (3, H, W)
        attr = gallery_item['attr'] # (3, H, W)
        h0 = gallery_item['h0'] # (H, W)
        y0 = gallery_item['y0']
        y_pred = gallery_item['y_pred']
        metrics_collection = gallery_item['metrics_collection']

        attrs = xt.sum_pixels_over_channels(attr, normalize='absmax_after_sum')
        attr_amplitude = np.max(np.abs(attr))

        th_soft_first, th_soft_last = metrics_collection[0]['thresholds'], metrics_collection[-1]['thresholds']
        fsetting = {'thresholds': th_soft_first,'targets': np.array([-2,-1,0,1,2]), }
        
        heatmap_magnitude = np.max(np.abs(fsetting['targets'])) 
        h = xai.five_band_stratification(attrs, setting=fsetting)/heatmap_magnitude
       
        fsetting['thresholds'] = th_soft_last
        h1 = xai.five_band_stratification(attrs, setting=fsetting)/heatmap_magnitude
         
        # print(h.shape, np.max(h), np.min(h))
        # print(h1.shape, np.max(h1), np.min(h1))

        row_items = {
            'x': (x.transpose(1,2,0), 'color_img',make_main_title(y_pred, y0, attr_amplitude)),
            'h0': (h0, 'heatmap', None),
            'attrs': (attrs, 'heatmap', None),
            'h': (h, 'heatmap',str(list(np.round(th_soft_first,5)))),
            'h1': (h1, 'heatmap', str(list(np.round(th_soft_last,5)))),
        }

        for j, (rowkey, rowitem) in enumerate(row_items.items()):
            fig.add_subplot(N_ROW, N_COL, N_COL*i+ (j+1))
            imgitem, colortype, this_title = rowitem
            if this_title is None: this_title = ''
            # print(rowkey, imgitem.shape)
            
            if colortype == 'color_img':
                plt.gca().imshow(imgitem) 
            elif colortype == 'heatmap':
                plt.gca().imshow(imgitem, cmap='bwr', vmin=-1., vmax=1.0)
            else:
                raise RuntimeError('invalide colortype.')
            if i>0:
                plt.gca().set_xticks([]); plt.gca().set_yticks([])   
            
            this_title = '%s\n%s'%(str(rowkey),str(this_title)) if i==0 else this_title
            plt.gca().set_title(this_title, fontsize=6)

    if ITEM_DIR is not None:
        plt.savefig(ITEM_DIR)
        plt.close()

def save_fig_by_gallery_label(gallery_label, gallery_items, ITEM_DIR):
    from pipeline.eval.eval_metrics import XAITranformation3c1c
    xt = XAITranformation3c1c()

    fig = plt.figure(figsize=(18,21))
    plt.tight_layout()
    N_COL, N_ROW = 6, 7
    for i, gallery_item in enumerate(gallery_items):
        x = gallery_item['x'] # (3, H, W)
        attr = gallery_item['attr'] # (3, H, W)
        h0 = gallery_item['h0'] # (H, W)
        y0 = gallery_item['y0']
        y_pred = gallery_item['y_pred']

        attr_amplitude = np.max(np.abs(attr))
        if np.all(attr==0):
            attr1 = attr
        else:
            attr1 = attr/attr_amplitude

        attrs = xt.sum_pixels_over_channels(attr, normalize='absmax_after_sum')
        # print('np.max(attr), np.min(attr):',np.max(attr), np.min(attr))

        row_items = {
            'x': (x.transpose(1,2,0), 'color_img', make_main_title(y_pred, y0, attr_amplitude)),
            'h0': (h0, 'heatmap', None),
            'attrR': (attr1[0], 'heatmap', None),
            'attrG': (attr1[1], 'heatmap', None),
            'attrB': (attr1[2], 'heatmap', None),
            'attrs': (attrs, 'heatmap', None)
        }

        for j, (rowkey, rowitem) in enumerate(row_items.items()):
            fig.add_subplot(N_ROW, N_COL, N_COL*i+ (j+1))
            imgitem, colortype, this_title = rowitem
            if this_title is None: this_title = ''
            # print(rowkey, imgitem.shape)
            
            if colortype == 'color_img':
                plt.gca().imshow(imgitem) 
            elif colortype == 'heatmap':
                plt.gca().imshow(imgitem, cmap='bwr', vmin=-1., vmax=1.0)
            else:
                raise RuntimeError('invalide colortype.')
            
            this_title = '%s\n%s'%(str(rowkey),str(this_title)) if i==0 else this_title
            plt.gca().set_title(this_title, fontsize=6)
            if i>0:
                plt.gca().set_xticks([]); plt.gca().set_yticks([])   
    
    if ITEM_DIR is not None:
        plt.savefig(ITEM_DIR)
        plt.close()

def unpack_and_pointwise_process_xai_data_control(config_data, display_decimal_precision=3, do_print=False, BRANCH_DATA=False,
    load_ext='xai',csv_ext='csv'):

    xai_mode = config_data['xai_mode']
    model_name = config_data['model_name']
    
    if not BRANCH_DATA:
        XAI_FOLDER = os.path.join('checkpoint', model_name, 'XAI_results')
        XAI_SAVEDIR = os.path.join(XAI_FOLDER, '%s_%s.%s'%(str(model_name), str(xai_mode), str(load_ext)))
        CSV_SAVEDIR = os.path.join(XAI_FOLDER, '%s_%s.%s'%(str(model_name), str(xai_mode), str(csv_ext)))
    else:
        XAI_FOLDER = os.path.join('checkpoint', model_name, '%s.%s'%(str(model_name),str(config_data['branch_name_label'])), 'XAI_results')
        XAI_SAVEDIR = os.path.join(XAI_FOLDER, '%s.%s_%s.%s'%(str(model_name), str(config_data['branch_name_label']), str(xai_mode), str(load_ext)))        
        CSV_SAVEDIR = os.path.join(XAI_FOLDER, '%s.%s_%s.%s'%(str(model_name), str(config_data['branch_name_label']), str(xai_mode), str(csv_ext)))

    print('unpack_and_pointwise_process_xai_data() %s'%(str(XAI_SAVEDIR)))
    
    evalpack = EvaluationPackage()
    evalpack = evalpack.load_pickled_data(XAI_SAVEDIR, tv=(1,0,100))

    MAIN_METRIC_NAMES = ['pixel_acc', 'recall', 'precision']
    ALT_MAIN_METRIC_NAMES = ['A','R','P']
    SUB_METRIC_TYPES = ['average', 'best', 'best_arg']
    ALT_SUB_METRIC_TYPES = ['mu', 'u', 'u1']

    general_table = {'id':[],'y0':[],'y_pred':[],'pred_is_correct':[]}
    for name in MAIN_METRIC_NAMES:
        for subtype in SUB_METRIC_TYPES:
            general_table['%s_dt_%s'%(str(name),str(subtype))] = []

    xai_result = getattr(evalpack, xai_mode)
    # print('  xai_mode:%s'%(str(xai_mode)))
    for identifier, data_point_package in xai_result.items():
        # print(identifier)
        labels = data_point_package['labels']
        metrics_collection = data_point_package['metrics_collection']

        pointwise_table = tabularize_data_point(metrics_collection)
        general_table['id'].append(identifier)
        general_table['y0'].append(labels['y0'])
        general_table['y_pred'].append(labels['y_pred'])
        general_table['pred_is_correct'].append(int(labels['pred_is_correct']))
        for name in MAIN_METRIC_NAMES:
            for subtype in SUB_METRIC_TYPES:
                if subtype == 'average':
                    this_value = np.mean(np.array(pointwise_table[name]))
                elif subtype == 'best':
                    this_value = np.max(np.array(pointwise_table[name]))
                elif subtype == 'best_arg':
                    this_value = np.argmax(np.array(pointwise_table[name]))
                general_table['%s_dt_%s'%(str(name),str(subtype))].append(this_value) # dt means data. To indicate pointwise data.

    general_table = pd.DataFrame(general_table)
    rename_dict = get_rename_dict(general_table, MAIN_METRIC_NAMES, ALT_MAIN_METRIC_NAMES, 
        SUB_METRIC_TYPES, ALT_SUB_METRIC_TYPES, display_decimal_precision=display_decimal_precision)

    general_table = general_table.rename(columns=rename_dict)
    if do_print: 
        print(general_table)
    # else:
    #     print('  Saving table!')
    general_table.to_csv(CSV_SAVEDIR)

def make_main_title(y_pred, y0, attr_amplitude):
    this_title = 'y:%s,y0:%s'%(str(y_pred),str(y0))
    this_title = this_title + '/A0:%s'%(str(attr_amplitude))
    return this_title

def get_rename_dict(df, MAIN_METRIC_NAMES, ALT_MAIN_METRIC_NAMES, 
    SUB_METRIC_TYPES, ALT_SUB_METRIC_TYPES, display_decimal_precision=3):
    pd.set_option("display.precision", display_decimal_precision)
    rename_dict = {}
    for name, alt_name in zip(MAIN_METRIC_NAMES, ALT_MAIN_METRIC_NAMES):
        for subtype, alt_subtype in zip(SUB_METRIC_TYPES, ALT_SUB_METRIC_TYPES):
            rename_dict['%s_dt_%s'%(str(name),str(subtype))] = '%s%s'%(str(alt_name),str(alt_subtype))

    # for x, xitem in rename_dict.items():
    #     print(x, xitem)
    return rename_dict

def unpack_and_pointwise_process_xai_data_for_roc_control(xai_setting, config_data,BRANCH_DATA=False,
    load_ext='xai',csv_ext='csv'):
    import pipeline.eval.xai_settings as xs 
    if load_ext=='xai': n_soft_steps = xs.XAI_SETTING_0001['n_soft_steps']
    elif load_ext == 'clamp.xai': n_soft_steps = xs.XAI_SETTING_0002['n_soft_steps']
    else: raise RuntimeError('invalid XAI setting.')

    VERBOSE = 0

    xai_mode = config_data['xai_mode']
    model_name = config_data['model_name']
    
    if not BRANCH_DATA:
        XAI_FOLDER = os.path.join('checkpoint', model_name, 'XAI_results')
        XAI_SAVEDIR = os.path.join(XAI_FOLDER, '%s_%s.%s'%(str(model_name), str(xai_mode), str(load_ext)))
        CSV_SAVEDIR = os.path.join(XAI_FOLDER, 'roc_%s_%s.%s'%(str(model_name), str(xai_mode), str(csv_ext)))
    else:
        XAI_FOLDER = os.path.join('checkpoint', model_name, '%s.%s'%(str(model_name),str(config_data['branch_name_label'])), 'XAI_results')
        XAI_SAVEDIR = os.path.join(XAI_FOLDER, '%s.%s_%s.%s'%(str(model_name), str(config_data['branch_name_label']), str(xai_mode), str(load_ext)))        
        CSV_SAVEDIR = os.path.join(XAI_FOLDER, 'roc_%s.%s_%s.%s'%(str(model_name), str(config_data['branch_name_label']), str(xai_mode), str(csv_ext)))

    print('unpack_and_pointwise_process_xai_data_for_roc() %s'%(str(XAI_SAVEDIR)))

    evalpack = EvaluationPackage()
    evalpack = evalpack.load_pickled_data(XAI_SAVEDIR, tv=(1,VERBOSE,250))
    # XAI_SAVEDIR example workflow1_0001.1_DeepLift.csv

    xai_result = getattr(evalpack, xai_mode)
    # print('  xai_mode:%s'%(str(xai_mode)))

    roc_data_by_threshold = {}
    for i_th in range(n_soft_steps): # collect item threshold by threshold
        if not i_th in roc_data_by_threshold:
            roc_data_by_threshold[i_th] = {'recall':[], 'FPR':[]}
        for identifier, data_point_package in xai_result.items():
            # print(identifier)
            labels = data_point_package['labels']
            metrics_by_threshold = data_point_package['metrics_collection'][i_th]

            roc_data_by_threshold[i_th]['recall'].append(metrics_by_threshold['recall'])
            roc_data_by_threshold[i_th]['FPR'].append(metrics_by_threshold['FPR'])
            # print(metrics_by_threshold)
            # raise Exception('DEBUG')

        roc_data_by_threshold[i_th]['recall'] = np.mean(roc_data_by_threshold[i_th]['recall'])
        roc_data_by_threshold[i_th]['FPR'] = np.mean(roc_data_by_threshold[i_th]['FPR'])
    df = rearrange_roc_data(roc_data_by_threshold)
    df.to_csv(CSV_SAVEDIR)

def rearrange_roc_data(roc_data_by_threshold):
    df = {    
        'recalls' : [],
        'FPRs' : []
    }
    for i_th, vals in roc_data_by_threshold.items():
        df['recalls'].append(vals['recall'])
        df['FPRs'].append(vals['FPR'])
    df = pd.DataFrame(df)
    return df