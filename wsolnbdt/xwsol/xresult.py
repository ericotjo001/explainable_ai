import os
from os.path import join as ospj
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_maxboxacc(RESULTS):
    """ For ILSVRC
    Adapted from evaluation.py BoxEvaluator(.compute()
    recall: maxboxacc(\delta)=max_{\tau}BoxAcc(\tau,\delta))where 
       \tau is scoremap threshold
       \ddlta is the iou_threshold
    """
    max_box_acc = []
    num_correct = RESULTS['num_correct']
    cnt = RESULTS['cnt']    
    iou_threshold_list = RESULTS['iou_threshold']

    for _THRESHOLD in iou_threshold_list:
        localization_accuracies = num_correct[_THRESHOLD] * 100. / float(cnt)
        max_box_acc.append(localization_accuracies.max())
    return max_box_acc

def get_operating_curve(RESULTS):
    num_correct = RESULTS['num_correct']
    cnt = RESULTS['cnt']    
    iou_threshold_list = RESULTS['iou_threshold']

    cam_curve = {}
    for _THRESHOLD in iou_threshold_list:
        localization_accuracies = num_correct[_THRESHOLD] * 100. / float(cnt)
        cam_curve[_THRESHOLD] = localization_accuracies
    return cam_curve

def collate_results(args_dict):
    print('collate_results()')
    print('** Assume the folder is named like <arch>_<method>_<layer>...')

    if args_dict['NBDT']:
        from xwsol.results_config_nbdt import result_config
    else:
        from xwsol.results_config import result_config

    RESULT_SAVE_FOLDER = ospj(args_dict['scoremap_root'],'_RESULTS') 
    if not os.path.exists(RESULT_SAVE_FOLDER):
        os.makedirs(RESULT_SAVE_FOLDER)

    cam_interval =  np.arange(0, 1, args_dict['cam_curve_interval'])

    font = {'size': 16}
    plt.rc('font', **font)

    print('==>preparing dataframe...')
    dataframe = {'method': [],'MaxBoxAcc2':[]}
    for iout in args_dict['iou_threshold_list']:
        dataframe[iout] = []

    print('\n==> preparing results....')
    for i,method_groups in enumerate(result_config['METHOD_LIST']):
        plt.figure(figsize=(16,6))
        axes_dict = {
            'ax1': plt.gcf().add_subplot(131),
            'ax2': plt.gcf().add_subplot(132),
            'ax3': plt.gcf().add_subplot(133),
        } 
        RESULT_IMG_DIR = ospj(RESULT_SAVE_FOLDER,'group%s'%(str(i)))
        for arch_method_layer in method_groups:
            aml = arch_method_layer.split('_')
            method = '_'.join(aml[1:])
            dataframe['method'].append(method)

            RESULT_DIR = ospj(args_dict['scoremap_root'], arch_method_layer,'score.result')
            RESULTS = joblib.load(RESULT_DIR)
            max_box_acc = compute_maxboxacc(RESULTS)
            max_box_acc2 = np.round(np.mean(max_box_acc),3)
            print('%-5s %-32s'%(str(max_box_acc2),str(max_box_acc)) ,method, )
            dataframe['MaxBoxAcc2'].append(max_box_acc2)

            assert(np.all(RESULTS['iou_threshold'] == args_dict['iou_threshold_list']))
            cam_curve = get_operating_curve(RESULTS)
            for i, iout in enumerate(RESULTS['iou_threshold']):
                dataframe[iout].append(max_box_acc[i])

                plt.sca(axes_dict['ax%s'%(str(i+1))])
                plt.plot(cam_interval, cam_curve[iout], label=method)
                plt.gca().set_title('iou threshold:%s'%(str(iout)))
                if i==0:
                    plt.gca().set_ylabel('BoxAcc')
                if i==1:
                    plt.gca().set_xlabel(r'operating threshold ($\tau$)')
        plt.legend(prop={'size': 11})
        plt.savefig(RESULT_IMG_DIR)

    RESULT_CSV_DIR = ospj(RESULT_SAVE_FOLDER, 'result.csv')
    df = pd.DataFrame(dataframe)
    df.to_csv(RESULT_CSV_DIR, index=False)

    # plt.show()


def get_cam_curve(arch_method_layer, scoremap_root):
    aml = arch_method_layer.split('_')
    method = '_'.join(aml[1:])

    RESULT_DIR = ospj(scoremap_root, arch_method_layer,'score.result')
    RESULTS = joblib.load(RESULT_DIR)

    cam_curve = get_operating_curve(RESULTS)
    return cam_curve, RESULTS

def compare_with_nbdt(args_dict):
    print('compare_with_nbdt')
    RESULT_FOLDER = ospj(args_dict['scoremap_root'],'_RESULTS') 
    RESULT_COMPARE_FOLDER = ospj(args_dict['scoremap_root_compare'],'_RESULTS') 

    # print(RESULT_FOLDER)
    # print(RESULT_COMPARE_FOLDER)

    from xwsol.results_config_compare import compare_config 
    font = {'size': 6}
    plt.rc('font', **font)
    plt.figure(figsize=(18,10))
    axes_dict = {
        'ax_30_saliency': plt.gcf().add_subplot(341),
        'ax_30_deeplift': plt.gcf().add_subplot(342),
        'ax_30_gradcam': plt.gcf().add_subplot(343),
        'ax_30_gbp': plt.gcf().add_subplot(344),
        'ax_50_saliency': plt.gcf().add_subplot(345),
        'ax_50_deeplift': plt.gcf().add_subplot(346),
        'ax_50_gradcam': plt.gcf().add_subplot(347),
        'ax_50_gbp': plt.gcf().add_subplot(348),
        'ax_70_saliency': plt.gcf().add_subplot(349),
        'ax_70_deeplift': plt.gcf().add_subplot(3,4,10),
        'ax_70_gradcam': plt.gcf().add_subplot(3,4,11),
        'ax_70_gbp': plt.gcf().add_subplot(3,4,12),
    } 

    cam_interval =  np.arange(0, 1, args_dict['cam_curve_interval'])
    n_methods = len(compare_config)

    display_y = {30:[-0.2,12], 50:[-0.2,3], 70:[-0.05,0.5]}
    colour_cycles = ['b','r','c','k','g']
    RESULT_IMG_DIR = os.path.join(RESULT_COMPARE_FOLDER,'global.png')
    for k, iout in enumerate([30,50,70]):
        for j, (method_name, method_group) in enumerate(compare_config.items()):
            normal_group = method_group['normal']
            nbdt_group = method_group['nbdt']

            for i,(arch_method_layer, arch_method_layer_nbdt)  in enumerate(zip(normal_group,nbdt_group)):
                cam_curve,RESULTS = get_cam_curve(arch_method_layer, args_dict['scoremap_root'])
                cam_curve_nbdt, RESULTS_nbdt = get_cam_curve(arch_method_layer_nbdt, args_dict['scoremap_root_compare'])

                plt.sca(axes_dict['ax_%s_%s'%(str(iout),str(method_name))])
                plt.plot(cam_interval, cam_curve[iout],  color='%s'%(str(colour_cycles[i])), linestyle='--', label=arch_method_layer)
                plt.plot(cam_interval, cam_curve_nbdt[iout], color='%s'%(str(colour_cycles[i])), linestyle='-', label=arch_method_layer_nbdt,)
                if k==0:
                    plt.gca().set_title('%s'%(str(method_name)))    
                if i==0 and j==0:
                    plt.gca().set_ylabel('iou threshold:%s\nBoxAcc'%(str(iout)))
                if i==2 and k==2:
                    plt.gca().set_xlabel(r'operating threshold ($\tau$)')

                plt.gca().set_ylim(display_y[iout])
                plt.legend()            

            # print(normal_group)
            # print(nbdt_group)

    plt.tight_layout()
    plt.savefig(RESULT_IMG_DIR)
    # plt.show()