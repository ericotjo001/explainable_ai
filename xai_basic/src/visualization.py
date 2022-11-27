from .utils import *

from .model import mabSPA
from .data import load_dataset_from_a_shard
from .objgen.random_simple_gen_implemented2 import ThreeClassesPyIOwithHeatmap 
from .objgen.random_simple_gen_implemented import TenClassesPyIOwithHeatmap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

XAI_METHODS = [
    'Saliency',
    'IntegratedGradients', 
    'DeepLift', 
    'GuidedBackprop', 
    'GuidedGradCam', 
    'Deconvolution',
    'GradientShap',
    'DeepLiftShap',
    'InputXGradient', 
    'mab',
]

def visualization_(dargs):
    print('visualization_')
    CKPT_FOLDER_DIR = dargs['CKPT_FOLDER_DIR']
    PROJECT_FOLDER_DIR = os.path.join(CKPT_FOLDER_DIR, dargs['PROJECT_NAME'])
    FIGURES_FOLDER_DIR =  os.path.join(PROJECT_FOLDER_DIR, 'figures')
    os.makedirs(FIGURES_FOLDER_DIR, exist_ok=True)
    
    marker_cycles = ['P', '8', 'X']
    color_scheme = np.linspace(0,1,len(XAI_METHODS))

    for model_name in dargs['model_names']:
        CSV_DIR = os.path.join(PROJECT_FOLDER_DIR, f'{model_name}_metric_aggreagate.csv' )
        df = pd.read_csv(CSV_DIR)
        # print(model_name,len(df))

        plt.figure(figsize=(9,9))
        for i,xai_method in enumerate(XAI_METHODS):
            df1 = df[df['xai_method']==xai_method]
            # print(xai_method, df1)
            
            color=(1-color_scheme[i],0,color_scheme[i],0.05)
            prec = df1['precision'].tolist()
            rec = df1['recall'].tolist()
            plt.scatter(prec,rec,
                s=3, c=color, marker=marker_cycles[i%3], )

        for i,xai_method in enumerate(XAI_METHODS):
            df1 = df[df['xai_method']==xai_method]
            # print(xai_method, df1)
            
            color=(1-color_scheme[i],0,color_scheme[i],)
            prec = df1['precision'].tolist()
            rec = df1['recall'].tolist()            
            plt.scatter(np.mean(prec), np.mean(rec),
                s=48, c=color,marker=marker_cycles[i%3], 
                linewidth=1., edgecolor='yellow',
                label=xai_method)
        plt.gca().set_xlim([-0.1,1.1])
        plt.gca().set_ylim([-0.1,1.1])
        plt.gca().set_xlabel('$P$')
        plt.gca().set_ylabel('$R$')
        plt.legend()

        FIG_DIR = os.path.join(FIGURES_FOLDER_DIR, f'recall_prec_{model_name}.png')
        plt.savefig(FIG_DIR)


def eval_and_gallery_(dargs):
    print('eval_and_gallery_')
    DIRS = manage_dir(dargs)
    FIGURES_FOLDER_DIR =  os.path.join(DIRS['PROJECT_FOLDER_DIR'], 'figures')
    GALLERY_DATA_DIR = os.path.join(FIGURES_FOLDER_DIR, f'gallery_{dargs["model_name"]}.data')
    os.makedirs(FIGURES_FOLDER_DIR, exist_ok=True)

    with open(DIRS['MODEL_INFO_DIR']) as f:
        model_info_dir = json.load(f)

    if not os.path.exists(GALLERY_DATA_DIR):
        print('data not yet collected, going for data collection...')
    else:
        print('visualizing data...')
        gallery = joblib.load(GALLERY_DATA_DIR)
        vis_gallery(gallery, dargs['model_name'], FIGURES_FOLDER_DIR)
        return

    if dargs['n_classes'] == 3:
        net = mabSPA().to(device=device)
    elif dargs['n_classes'] == 10:
        net = mabSPA(fc_output_c=10).to(device=device)
    else:
        raise NotImplementedError()    
    checkpoint = torch.load(DIRS['MODEL_DIR'])

    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()   

    n_total, n_correct = 0,0
    gallery = {
        'correct': [], 'n_correct':0,
        'wrong': [], 'n_wrong':0
    }

    def collect_gallery_item_by_index(k, x,y0,y_pred,h0,h1):
        img = x[k].clone().cpu().numpy().transpose(1,2,0)
        label, pred = y0[k].item(), y_pred[k].item()
        heatmap_gt = (h0[k]==1)*0.4 + (h0[k]==2)*0.9
        attr = (h1[k]==1)*0.4 + (h1[k]==2)*0.9  
        return (img, label, pred, heatmap_gt, attr)

    with torch.no_grad():
        shardlist = os.listdir(DIRS['SHARD_TEST_FOLDER_DIR'])
        for ns,SHARD_NAME in enumerate(shardlist):
            if not SHARD_NAME[-6:] == '.shard': continue

            SHARD_DIR = os.path.join(DIRS['SHARD_TEST_FOLDER_DIR'], str(SHARD_NAME), )
            print('processing SHARD_DIR %s'%(str(SHARD_DIR)))
            testdataset = load_dataset_from_a_shard(SHARD_DIR, reshape_size=None)
            if dargs['n_classes'] == 3:
                testdataset = ThreeClassesPyIOwithHeatmap(testdataset.x,testdataset.y,testdataset.h)
            elif dargs['n_classes'] == 10:
                testdataset = TenClassesPyIOwithHeatmap(testdataset.x,testdataset.y, testdataset.h)
            else:
                raise NotImplementedError()
            testloader = DataLoader(testdataset,batch_size=16) 

            for i,(x,y0, h0) in enumerate(testloader):            
                x = x.to(torch.float).to(device=device)
                y0 = y0.to(torch.long).to(device=device)
                h0 = h0.clone().cpu().numpy()
                
                b,c,h_,w_ = x.shape

                y, h = net(x)
                y_pred = torch.argmax(y,dim=1)            
                h1 = torch.argmax(h[:,:,:h_,:w_], dim=1)
                h1 = h1.clone().detach().cpu().numpy()


                n_correct += torch.sum(y0==y_pred).item()
                n_total += len(y0)

                if gallery['n_correct'] < 20: # let's collect at least 20
                    _correct_indices = torch.where((y0==y_pred))[0].clone().cpu().numpy() 
                    for k in _correct_indices:
                        # print(x[k].shape, y0[k], y_pred[k], h0[k].shape, h1[k].shape) # like torch.Size([3, 256, 256]) tensor(0, device='cuda:0') tensor(0, device='cuda:0') (256, 256) (256, 256)
                        gallery['correct'].append(collect_gallery_item_by_index(k, x,y0,y_pred,h0,h1))
                        gallery['n_correct'] += 1

                if gallery['n_wrong'] < 10: # let's collect at least 10
                    _false_indices = torch.where((y0!=y_pred))[0].clone().cpu().numpy()
                    for k in _false_indices:
                        gallery['wrong'].append(collect_gallery_item_by_index(k,x,y0,y_pred,h0,h1))
                        gallery['n_wrong'] += 1                   
                
        test_acc = n_correct/n_total
        print(f'test_acc: {n_correct}/{n_total} = {test_acc}',)
        model_info_dir.update({'test_acc': test_acc})
        with open(DIRS['MODEL_INFO_DIR'], 'w') as json_file:
            json.dump(model_info_dir, json_file, indent=4, sort_keys=True)

        joblib.dump(gallery, GALLERY_DATA_DIR)

        print('data collection complete. run the code once again to visualize the data')


def vis_gallery(gallery, model_name, FIGURES_FOLDER_DIR):
    def off_ticks():
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
    max_n_rows = 7 # per figure
    font = {'size' : 7}
    figsize = (3,8)
    titlefontdict = {'fontsize':7}
    import matplotlib
    matplotlib.rc('font', **font)

    def saveplot(gallerylabel):
        j = 0
        count = 1
        plt.figure(figsize=figsize)
        for i, (img, label, pred, heatmap_gt, attr) in enumerate(gallery[gallerylabel]):
            plt.gcf().add_subplot(max_n_rows,3, 3* j + 1)
            plt.gca().imshow(img)
            plt.gca().set_title(f'label/pred:{label}/{pred}', fontdict=titlefontdict)
            if j>0: off_ticks()
            
            plt.gcf().add_subplot(max_n_rows,3, 3* j + 2)
            plt.gca().imshow(heatmap_gt ,cmap='bwr', vmin=-1., vmax=1.)
            if j == 0:  plt.gca().set_title('groundtruth',fontdict=titlefontdict)
            off_ticks()
            
            plt.gcf().add_subplot(max_n_rows,3, 3* j + 3)
            plt.gca().imshow(attr, cmap='bwr', vmin=-1., vmax=1.)
            off_ticks()
            if j == 0:  plt.gca().set_title('heatmap pred',fontdict=titlefontdict)
            plt.tight_layout()
            j+=1

            if j >= max_n_rows:
                # plt.show()
                plt.savefig(os.path.join(FIGURES_FOLDER_DIR, f'gallery_{model_name}_{gallerylabel}_{count}.png'))
                j = 0
                count += 1
                plt.figure(figsize=figsize)

        if j>0:
            plt.savefig(os.path.join(FIGURES_FOLDER_DIR, f'gallery_{model_name}_{gallerylabel}_{count}.png')) 
            # plt.show()
    saveplot('correct')
    saveplot('wrong')
    return