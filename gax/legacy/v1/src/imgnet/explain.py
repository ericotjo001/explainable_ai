import os, random, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from skimage.transform import resize

from .train import pretrained_model_selection
from .utils import manage_directories, FastPickleClient
from ..utils import sp

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from captum.attr import Saliency, InputXGradient, LayerGradCam
from captum.attr import Deconvolution, GuidedBackprop, DeepLift

def manage_xai_result_dir(DIRS, args):
    SPLIT = args['split']    

    XAI_COLLECT_FOLDER = os.path.join(DIRS['PROJECT_DIR'], 'xai_collect')
    if not os.path.exists(XAI_COLLECT_FOLDER): os.mkdir(XAI_COLLECT_FOLDER)
    XAI_COLLECT_DIR =  os.path.join(XAI_COLLECT_FOLDER, 'xai_collect.dictionary')
    DIRS['XAI_COLLECT_FOLDER'] = XAI_COLLECT_FOLDER
    DIRS['XAI_COLLECT_DIR'] = XAI_COLLECT_DIR
    
    # IMG_FOLDER_DIR = os.path.join(DIRS['DATA_DIR'], SPLIT, args['label'])
    # DIRS['IMG_FOLDER_DIR'] = IMG_FOLDER_DIR
    return DIRS


class GAX(object):
    def __init__(self, args):
        super(GAX, self).__init__()
        self.args = args
        self.DIRS = manage_directories(args)
        self.save_manager = FastPickleClient()

    def get_attribution_by_method(self, net, x, y0, **args):
        assert(x.shape[0]==1) # process ONE at a time
        x.requires_grad = True
        method = args['method']

        y_attr = net(x)
        y_attr = torch.argmax(y_attr,dim=1)
        if method=='Saliency':
            attr = Saliency(net).attribute(x, target=y_attr, abs=False)
            # print(attr.shape) # torch.Size([1, 3, 256, 256])
        elif method=='InputXGradient':
            attr = InputXGradient(net).attribute(x, target=y_attr)
        elif method =='LayerGradCam':
            img_size = (args['img_size'],args['img_size']) 
            if args['model'] == 'resnet34':
                attr = LayerGradCam(net, layer=getattr(net.backbone,'conv1')).attribute(x, target=y_attr)
            elif args['model'] =='alexnet':
                attr = LayerGradCam(net, layer=net.backbone.features[0]).attribute(x, target=y_attr)
            else:
                raise NotImplementedError()
            
            attr = resize(attr[0].clone().detach().cpu().numpy().transpose(2,1,0), img_size + (3,))
            attr = torch.from_numpy(np.array([attr.transpose(2,1,0)])).to(device=device)
            # see shape in Saliency
        elif method=='Deconvolution':
            attr = Deconvolution(net).attribute(x, target=y_attr)
        elif method=='GuidedBackprop':
            attr = GuidedBackprop(net).attribute(x, target=y_attr)
        elif method =='DeepLift':
            attr = DeepLift(net).attribute(x, target=y_attr)
        else:
            raise RuntimeError('Invalid method')
        
        return attr

    def display_co_score_boxplot_for_existing_methods(self, dir_method=None):
        print('display_co_score_boxplot_for_existing_methods()')
        args = self.args
        DIRS = self.DIRS

        if dir_method is None:
            DIRS = manage_xai_result_dir(DIRS,args) 
        else:
            DIRS = dir_method(DIRS, args)
        XAI_COLLECT = self.save_manager.load_pickled_data(DIRS['XAI_COLLECT_DIR'], tv=(0,0,100), text=None)

        from collections import defaultdict
        font = {'size': 8} # 'weight': 'bold'
        plt.rc('font', **font)

        for this_submethod in ['sum','mult']:
            SAVE_IMG_DIR = os.path.join(DIRS['XAI_COLLECT_FOLDER'], 'co_boxplot.%s.jpeg'%(this_submethod))
            boxplot_collections = defaultdict(list)
            for collection_keys, data in XAI_COLLECT.items():
                _, _, method, submethod = collection_keys
                if not submethod == this_submethod: continue
                iscorrect = data['labelCorrect']
                co_score = data['co_score']

                boxplot_key = '%s.%s'%(str(method),str(iscorrect))
                boxplot_collections[boxplot_key].append(co_score)

            plt.figure(figsize=(int(len(boxplot_collections)/2),5))
            plt.boxplot([co_score for _,co_score in boxplot_collections.items()],
                flierprops={'marker':'.', 'markersize':1,} 
            )
            plt.xticks( np.array(range(len(boxplot_collections)+1))-0.5,
                [''] + [co_score for co_score in boxplot_collections], rotation=60)
            plt.xlim([0.5,None])
            plt.ylabel('CO score')
            plt.tight_layout()
            plt.savefig(SAVE_IMG_DIR)
        plt.close()

    def display_co_score_for_existing_methods(self, dir_method=None):
        print('display_co_score_for_existing_methods()')
        args = self.args
        DIRS = self.DIRS 
        METHOD = args['method']

        if dir_method is None:
            DIRS = manage_xai_result_dir(DIRS,args) 
        else:
            DIRS = dir_method(DIRS, args)
        XAI_COLLECT = self.save_manager.load_pickled_data(DIRS['XAI_COLLECT_DIR'], tv=(0,0,100), text=None)

        for submethod in ['sum','mult']:
            SAVE_IMG_DIR = os.path.join(DIRS['XAI_COLLECT_FOLDER'], 'co_hist.%s.%s.jpeg'%(str(METHOD),str(submethod)))

            co_correct, co_wrong = [],[]
            for collection_keys, data in XAI_COLLECT.items():
                CONDITIONS = [
                    collection_keys[2] == METHOD ,
                    collection_keys[3] == submethod , 
                ]

                if np.all(CONDITIONS):
                    # print(collection_keys, data)
                    if data['labelCorrect'] == 'correct':
                        co_correct.append(data['co_score'])
                    elif data['labelCorrect'] == 'wrong':
                        co_wrong.append(data['co_score'])

            plt.figure()
            plt.gcf().add_subplot(111)
            plt.gca().hist(co_correct, alpha=0.2, label='correct')
            plt.gca().hist(co_wrong, alpha=0.1, label='wrong')
            plt.gca().set_xlabel('CO score')
            plt.legend()
            plt.title('histogram for %s [%s]'%(str(METHOD),str(submethod)))
            plt.savefig(SAVE_IMG_DIR)
            plt.close()

    def collect_co_score_for_existing_methods(self, dir_method=None):
        print('collect_co_score_for_existing_methods()')

        args = self.args
        DIRS = self.DIRS
        img_size = (args['img_size'],args['img_size']) 

        SPLIT = args['split']
        METHOD = args['method']

        print('collect_co_score_for_existing_methods() with %s, split:%s'%(str(METHOD),str(SPLIT)))

        def load_or_create_result_dictionary(XAI_COLLECT_DIR):
            if os.path.exists(XAI_COLLECT_DIR):
                print('Loading XAI_COLLECT!')
                XAI_COLLECT = self.save_manager.load_pickled_data(DIRS['XAI_COLLECT_DIR'], tv=(0,0,100), text=None)
            else:
                print('New XAI_COLLECT...') 
                XAI_COLLECT = {} # see compute_co_score_by_method()
            return XAI_COLLECT

        net = pretrained_model_selection(args['model'], eval=True, device=device)
        if dir_method is None:
            DIRS = manage_xai_result_dir(DIRS,args) 
        else:
            DIRS = dir_method(DIRS, args)
        XAI_COLLECT = load_or_create_result_dictionary(DIRS['XAI_COLLECT_DIR'])

        normalize = transforms.Compose([
            transforms.Resize((args['img_size'],args['img_size'])),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])

        start = time.time()
        if SPLIT == 'val':
            n_total = 50000
            from .imagenet_utils import ImageNetValidation
            imgn = ImageNetValidation(MAIN_DATA_DIR=DIRS['DATA_DIR'])
            all_indices = np.array(range(n_total))
            random.shuffle(all_indices)
            for i, this_index in enumerate(all_indices):
                img, y0, label_text, img_name = imgn.get_val_data_by_index(i, as_pytorch_tensor=True)

                x = normalize(img).to(device=device)
                y0 = y0.to(device=device)
                attr = self.get_attribution_by_method(net, x, y0, **args)
                
                for submethod in ['sum','mult']:
                    XAI_COLLECT, alrd_registered = self.compute_co_score_by_method(net, x, y0, attr, 
                        XAI_COLLECT, method=METHOD, submethod=submethod,
                        img_name=img_name, split=SPLIT)

                    if (i+1)%4==0 or (i+1)==n_total:
                        update_str = '%s/%s | %s | submethod:%s | alrd_registered?:%s | n=%s'%(str(i+1),str(n_total), str(img_name),
                            str(submethod), str(alrd_registered),str(len(XAI_COLLECT)))
                        print('%-96s'%(str(update_str)),end='\r')
                
                if args['n_debug_imagenet']>0:
                    if i>args['n_debug_imagenet']: break

        elif SPLIT == 'train': 
            raise RuntimeError('Not Implemented')
        print('')
        self.save_manager.pickle_data(XAI_COLLECT, DIRS['XAI_COLLECT_DIR'], tv=(0,0,100), text=None)
        end = time.time()
        elapsed = end - start
        print(' time taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))))


    def compute_co_score_by_method(self, net, x, y0, attr, 
                            XAI_COLLECT, method, submethod,
                            img_name, split):
        # attr: heatmap derived from the predicted label (not groundtruth).

        alrd_registered = False
        collection_keys = (img_name,split, method, submethod,)
        if collection_keys in XAI_COLLECT:
            alrd_registered = True
            return XAI_COLLECT, alrd_registered

        attr = attr.detach()
        attr_norm = torch.max(torch.abs(attr))
        attr = attr/attr_norm

        # for modifier_method in ['sum', 'mult']:
        with torch.no_grad():
            y = net(x)
            n_class = y.shape[1]
            if submethod=='sum':
                y_aug = net(x + attr)
            elif submethod =='mult':
                y_aug = net(x*attr)

            y_pred = torch.argmax(y,dim=1)[0].item()
            y0_label = int(y0[0].item())

            # score
            score_constants = torch.zeros_like(y) - 1./(n_class-1)
            score_constants[0,y0_label] = 1.
            co_score = (y_aug -y)*score_constants
            co_score = torch.sum(co_score).item() # machine interpretability credit score

            labelCorrect = 'correct' if y_pred==y0_label else 'wrong'                
            XAI_COLLECT[collection_keys] = {
                'co_score': co_score, 'labelCorrect': labelCorrect, 'y0':y0_label,'y_pred': y_pred
            }

            
        return XAI_COLLECT, alrd_registered


    def gax(self):
        print('gax')
        from .model import Generator

        args = self.args
        DIRS = self.DIRS
        
        img_size = (args['img_size'],args['img_size']) 
        OPTIM_IMG_FOLDER = os.path.join(DIRS['PROJECT_DIR'],'gax_images')
        if not os.path.exists(OPTIM_IMG_FOLDER): os.mkdir(OPTIM_IMG_FOLDER)

        from .imagenet_utils import ImageNetValidation
        imgn = ImageNetValidation(MAIN_DATA_DIR=DIRS['DATA_DIR'])       

        from .train import pretrained_model_selection
        net = pretrained_model_selection(args['model'], eval=True,device=device)

        resize = transforms.Compose([transforms.Resize((256,256)),])
        normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])

        def one_step(netG,x,y, y_pred, n_class, submethod, ):
            # x is raw image (at most resized), pixel range is 0 to 1
            epsilon = 1e-4
            netG.train()
            netG.zero_grad()
            attr_op = netG(x)

            if submethod == 'sum':
                yg = net(x+attr_op)
            elif submethod == 'mult':
                yg = net(x*attr_op)
            else:
                raise RuntimeError('invalid submethod')
            
            score_constants = torch.zeros_like(y) - 1./(n_class-1)
            score_constants[0,y_pred] = 1.
            co_score = (yg -y)*score_constants
            co_score = torch.sum(co_score)
            loss = -co_score + args['similarity_loss_factor']/torch.mean((attr_op-x+epsilon)**2/(x+epsilon))
            loss.backward()
            optimizerG.step()
            return netG, co_score        

        if args['img_index'] is not None:
            max_count = 1
        else:
            assert(args['first_n_correct']>0)
            max_count = args['first_n_correct']

        index_counter = 0
        submethod = args['submethod']
        n_correct=0
        while True:
            if args['img_index'] is not None:
                img_index = args['img_index']
            else:
                img_index  = index_counter
            img_name = imgn.VAL_IMG_LIST[index_counter].split('.')[0]

            try:
                img, y0, label_text, img_name = imgn.get_val_data_by_index(index_counter, as_pytorch_tensor=True)
            except:
                print('failed to open image at img_index:', img_index)
                continue

            with torch.no_grad():
                x_pre = resize(img).to(device=device)
                x = normalize(x_pre.clone())
                y = net(x)
                y_pred = torch.argmax(y,dim=1)[0].item()
                correct_prediction = y_pred==y0[0].item()

                if not correct_prediction:
                    print('prediction is incorrect at index ', img_index)
                    if args['img_index'] is not None:
                        return False
                    else:
                        index_counter+=1; continue
                else:
                    n_correct+=1
                    
            netG = Generator(img_size=img_size)
            netG = netG.to(device=device)
            optimizerG = optim.Adam(netG.parameters(), lr=args['gax_learning_rate'], betas=(0.9, 0.999), )#
                
            n_class = y.shape[1]
            co_score = -np.inf
            OPTIM_IMG_DIR = os.path.join(OPTIM_IMG_FOLDER,'op.%s.%s.%s'%(str(img_name),str(args['split']),str(submethod)))
            OPTIM_SCORE_DIR = os.path.join(OPTIM_IMG_FOLDER,'op.%s.%s.%s.COS'%(str(img_name),str(args['split']),str(submethod)))
            # print('\ntraining for method:GAX submethod:%s'%(str(submethod)))
            imgs, co_scores = [],[]
            
            if args['target_co']>0:
                args['n_iter'] = 0

            i = 0
            while True:                
                # train normally
                netG, co_score = one_step(netG,x_pre ,y, y_pred, n_class, submethod=args['submethod'])
                co_scores.append(co_score.item())

                if i==0 or (i+1)%4==0 or (i+1)==args['n_iter']:
                    main_update_str = '%s/%s oc:%.8f'%(str(i+1),str(args['n_iter']),co_score.item())
                    print('%-64s'%(str(main_update_str)),end='\r')
                    
                netG.eval()
                attrg = netG(x_pre)
                attrg = attrg.clone().detach().cpu().numpy()[0].transpose(2,1,0)
                imgs.append(attrg)                    

                i+=1
                if args['target_co']==0:
                    if i>=args['n_iter']: break
                else:
                    if co_score.item()>args['target_co']:
                        print('img_index:%s target achieved at iter %s. co_score:%s submethod:%s'%(str(img_index),str(i),str(co_score.item()),str(submethod)))
                        break

            data = np.array(imgs)
            np.save(OPTIM_IMG_DIR, data)
            np.save(OPTIM_SCORE_DIR, np.array(co_scores))

            index_counter+=1
            if n_correct>=max_count: break

        return True

    def mass_plot_co_scores(self):
        print('mass_plot_co_scores')
        args = self.args
        DIRS = self.DIRS

        PROJECT_DIRS = [os.path.join(DIRS['CHECKPOINT_DIR'], x) for x in args['PROJECT_IDs']]
        colors = ['b','r']
        plt.figure()
        plt.gcf().add_subplot(111)
        for i, (this_project_id, THIS_PROJECT_DIR) in enumerate(zip(args['PROJECT_IDs'],PROJECT_DIRS)):
            OPTIM_IMG_FOLDER = os.path.join(THIS_PROJECT_DIR,'gax_images')
            list_of_scores_dir = [x for x in os.listdir(OPTIM_IMG_FOLDER) if 'COS' in x and 'val' in x]
            for j, OPTIM_SCORE_DIR in enumerate(list_of_scores_dir):
                co_score = np.load(os.path.join(OPTIM_IMG_FOLDER, OPTIM_SCORE_DIR))
                if j==0:
                    plt.gca().plot(co_score, c=colors[i], linewidth=0.1, label=this_project_id)
                else:
                    plt.gca().plot(co_score, c=colors[i], linewidth=0.1)
                plt.gca().set_xlim([0,300])
        plt.gca().set_ylabel('CO score')
        plt.gca().set_xlabel('iter')
        plt.legend()
        plt.show()

    def display_gax_optimized_images(self):
        print('display_gax_optimized_images()')

        args = self.args
        DIRS = self.DIRS
        img_size = (args['img_size'],args['img_size']) 
        OPTIM_IMG_FOLDER = os.path.join(DIRS['PROJECT_DIR'],'gax_images')

        from .imagenet_utils import ImageNetValidation
        imgn = ImageNetValidation(MAIN_DATA_DIR=DIRS['DATA_DIR'])       
        img_name = imgn.VAL_IMG_LIST[args['img_index']].split('.')[0]
        img, y0, label_text, img_name = imgn.get_val_data_by_index(args['img_index'], as_pytorch_tensor=True)
        x = img
        
        print(img_name)

        OPTIM_IMG_DIR = os.path.join(OPTIM_IMG_FOLDER,'op.%s.%s.%s.npy'%(str(img_name),str(args['split']),str(args['submethod'])))
        OPTIM_SCORE_DIR = os.path.join(OPTIM_IMG_FOLDER,'op.%s.%s.%s.COS.npy'%(str(img_name),str(args['split']),str(args['submethod'])))

        data = np.load(OPTIM_IMG_DIR)
        co_scores = np.load(OPTIM_SCORE_DIR)
        D ,W, H, C = data.shape

        plt.figure()
        plt.imshow(img.clone().detach().cpu().numpy()[0].transpose(1,2,0))
        plt.gca().set_xlabel(label_text)

        fig = plt.figure(figsize=(12,4))
        ax = fig.add_subplot(161)
        self.off_ticks(yticks=True,xticks=True)
        ax2 = fig.add_subplot(162)
        self.off_ticks()
        ax3 = fig.add_subplot(163)
        self.off_ticks()

        change = np.zeros(shape=data.shape)
        change[:-1] = np.abs(data[1:]-data[:-1])
        ax4 = fig.add_subplot(164)
        self.off_ticks()
        ax5 = fig.add_subplot(165)
        self.off_ticks()
        ax6 = fig.add_subplot(166)
        self.off_ticks()

        plt.tight_layout()

        ip = InteractivePlot(data,[ax,ax2,ax3,ax4, ax5,ax6], fig, 
            change=change, co_scores=co_scores, x=x, y0=y0, img_size=args['img_size'] )
        slider_depth = ip.setup_slider(D)
        ip.update(0,init=True)
        slider_depth.on_changed(ip.update)
        plt.show()

    def off_ticks(self,xticks=False, yticks=False):
        if not xticks:
            plt.gca().set_xticks([])
        if not yticks:
            plt.gca().set_yticks([])


class InteractivePlot(object):
    def __init__(self, data, axes, fig, **kwargs):
        super(InteractivePlot, self).__init__()
        self.kwargs = kwargs

        self.cb = None
        self.data = data
        self.fig = fig

        for i,ax in enumerate(axes):
            label = str(i+1) if i+1>1 else ''
            setattr(self,'ax%s'%(str(label)), axes[i])

        img_size = kwargs['img_size']
        self.img = kwargs['x'].clone().detach().cpu().numpy()[0].transpose(1,2,0)
        self.y0 = kwargs['y0'].clone().detach().cpu().numpy()[0]
        self.img_overlay = resize(self.img, (img_size,img_size))

    def update(self, val, init=False):
        # val is automatically slider_depth.val
        if init:
            current_depth = 0
        else:
            current_depth = int(val)

        x = self.data[current_depth,:,:,:].transpose(1,0,2)
        amax = np.max(np.abs(x))
        for i,name in enumerate(['',2,3]):
            getattr(self,'ax%s'%(str(name))).imshow(x[:,:,i], cmap='bwr', vmin=-amax, vmax=amax) # , vmin=-1.,vmax=1.
            # getattr(self,'ax%s'%(str(name))).imshow(self.img,alpha=0.2)
            getattr(self,'ax%s'%(str(name))).imshow(self.img_overlay, alpha=0.2)

        self.ax2.set_xlabel('abs max:%s'%(str(amax)))
        self.ax3.set_xlabel('step: %s score:%s'%(str(current_depth),
            str(np.round(self.kwargs['co_scores'][current_depth],3))))
        
        c = self.kwargs['change'][current_depth,:,:,:].transpose(1,0,2)
        self.ax4.imshow(c[:,:,0], cmap='Reds', vmin=0.)
        self.ax5.imshow(c[:,:,1], cmap='Reds', vmin=0.)
        self.ax6.imshow(c[:,:,2], cmap='Reds', vmin=0.)
                
        if not init: self.fig.canvas.draw_idle()
 
    def setup_slider(self,D):
        axcolor = 'lightgoldenrodyellow'
        posx, posy, widthx_fraction, widthy_fraction = 0.1, .93, 0.3, 0.05
        ax_depth = plt.axes([posx, posy, widthx_fraction, widthy_fraction], facecolor=axcolor)
        slider_depth = Slider(ax_depth, 'Time Step', 0, D-1, valinit=0, valstep=1)
        return slider_depth 