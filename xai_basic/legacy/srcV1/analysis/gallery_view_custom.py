import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
For analysis purpose, modified versions of functions to view the gallery of XAI results
  are copied here and customized. 
"""

def save_fig_by_gallery_label_banded_heatmaps(gallery_label, gallery_items, ITEM_DIR,
    CLAMP_VMIN=-0.1,CLAMP_VMAX=0.1):
    from pipeline.eval.eval_metrics import XAITranformation3c1c
    from pipeline.eval.eval_metrics import FiveBandXAIMetric
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

        th0, th1 = metrics_collection[0]['thresholds'], metrics_collection[-1]['thresholds']
        fsetting = {'thresholds': th0,'targets': np.array([-2,-1,0,1,2]), }
        heatmap_magnitude = np.max(np.abs(fsetting['targets'])) 
        h = xai.five_band_stratification(attrs, setting=fsetting)/heatmap_magnitude
        h0a = xai.five_band_stratification(h0, setting=fsetting)/heatmap_magnitude
        fsetting['thresholds'] = th1
        h1 = xai.five_band_stratification(attrs, setting=fsetting)/heatmap_magnitude
        h0b = xai.five_band_stratification(h0, setting=fsetting)/heatmap_magnitude
        
        # print(h.shape, np.max(h), np.min(h))
        # print(h1.shape, np.max(h1), np.min(h1))

        row_items = {
            'x': (x.transpose(1,2,0), 'color_img',make_main_title(y_pred, y0, attr_amplitude)),
            'h0': (h0, 'heatmap', None),
            'attrs': (attrs, 'heatmap_clamp', None),
            'h0a': (h0a, 'heatmap',str(list(np.round(th0,5)))),
            'h': (h, 'heatmap_clamp', None),
            'h0b': (h0b, 'heatmap', str(list(np.round(th1,5)))),
            'h1': (h1, 'heatmap_clamp', None),
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
            elif colortype == 'heatmap_clamp':
                plt.gca().imshow(imgitem, cmap='bwr', vmin=CLAMP_VMIN, vmax=CLAMP_VMAX)
            else:
                raise RuntimeError('invalide colortype.')
            if i>0:
                plt.gca().set_xticks([]); plt.gca().set_yticks([])   
            
            this_title = '%s\n%s'%(str(rowkey),str(this_title)) if i==0 else this_title
            plt.gca().set_title(this_title, fontsize=6)

    if ITEM_DIR is not None:
        plt.savefig(ITEM_DIR)
        plt.close()

def save_fig_by_gallery_label(gallery_label, gallery_items, ITEM_DIR,
    CLAMP_VMIN=-0.1,CLAMP_VMAX=0.1):
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
            'attrR': (attr1[0], 'heatmap_clamp', None),
            'attrG': (attr1[1], 'heatmap_clamp', None),
            'attrB': (attr1[2], 'heatmap_clamp', None),
            'attrs': (attrs, 'heatmap_clamp', None)
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
            elif colortype == 'heatmap_clamp':
                plt.gca().imshow(imgitem, cmap='bwr', vmin=CLAMP_VMIN, vmax=CLAMP_VMAX)
            else:
                raise RuntimeError('invalide colortype.')
            
            this_title = '%s\n%s'%(str(rowkey),str(this_title)) if i==0 else this_title
            plt.gca().set_title(this_title, fontsize=6)
            if i>0:
                plt.gca().set_xticks([]); plt.gca().set_yticks([])   
    
    if ITEM_DIR is not None:
        plt.savefig(ITEM_DIR)
        plt.close()

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