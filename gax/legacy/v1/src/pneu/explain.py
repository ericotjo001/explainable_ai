import os, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from skimage.transform import resize

from .utils import FastPickleClient
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from captum.attr import Saliency, InputXGradient, LayerGradCam
from captum.attr import Deconvolution, GuidedBackprop, DeepLift

"""
CO Score: confidence optimization score
"""

def manage_xai_result_dir(DIRS, args):
    SPLIT = args['split']    

    IMG_FOLDER_DIR = os.path.join(DIRS['DATA_DIR'], SPLIT, args['label'])
    XAI_COLLECT_FOLDER = os.path.join(DIRS['PROJECT_DIR'], 'xai_collect')
    if not os.path.exists(XAI_COLLECT_FOLDER): os.mkdir(XAI_COLLECT_FOLDER)
    XAI_COLLECT_DIR =  os.path.join(XAI_COLLECT_FOLDER, 'xai_collect.dictionary')
    DIRS['XAI_COLLECT_FOLDER'] = XAI_COLLECT_FOLDER
    DIRS['XAI_COLLECT_DIR'] = XAI_COLLECT_DIR
    DIRS['IMG_FOLDER_DIR'] = IMG_FOLDER_DIR
    return DIRS


def get_label(args):
    LABEL = args['label']
    assert(LABEL in ['NORMAL', 'PNEUMONIA'])
    LABEL_INDEX = 0
    if LABEL =='PNEUMONIA': LABEL_INDEX = 1
    return LABEL, LABEL_INDEX

class GAX(object):
    def __init__(self, args):
        super(GAX, self).__init__()
        self.args = args

        from .utils import manage_directories
        self.DIRS = manage_directories(args)
        self.save_manager = FastPickleClient()

    def off_ticks(self,xticks=False, yticks=False):
        if not xticks:
            plt.gca().set_xticks([])
        if not yticks:
            plt.gca().set_yticks([])

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

    def gax(self):
        from .model import Generator
        
        args = self.args
        DIRS = self.DIRS
        DIRS = manage_xai_result_dir(DIRS, args)

        img_size = (args['img_size'],args['img_size']) 
        OPTIM_IMG_FOLDER = os.path.join(DIRS['PROJECT_DIR'],'gax_images')
        if not os.path.exists(OPTIM_IMG_FOLDER): os.mkdir(OPTIM_IMG_FOLDER)

        LABEL, LABEL_INDEX = get_label(args)
        
        net = self.load_model(DIRS)  

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
        
        IMG_FOLDER_DIR = DIRS['IMG_FOLDER_DIR']

        if args['img_name'] is None:
            IMG_NAMELIST = os.listdir(IMG_FOLDER_DIR)

        index_counter, n_correct = 0, 0
        submethod = args['submethod']
        while True:                
            if args['img_name'] is not None:
                max_count = 1
                img_name = args['img_name']
            else:
                assert(args['first_n_correct']>0)
                max_count = args['first_n_correct']
                img_name = IMG_NAMELIST[index_counter]

            IMG_DIR = os.path.join(IMG_FOLDER_DIR, img_name)
            x, y0 = get_img_label_pair(IMG_DIR=IMG_DIR, LABEL=LABEL, img_size=img_size, device=device)
            # print(x.shape,y0) # torch.Size([1, 3, 256, 256]) tensor([0], device='cuda:0')
            with torch.no_grad():
                y = net(x)
                net.eval()
                y_pred = torch.argmax(y,dim=1)[0].item()
                correct_prediction = y_pred==y0[0].item()
                if not correct_prediction:
                    print('prediction is incorrect at index ', img_name)
                    if args['img_name'] is not None:
                        return False
                    else:
                        index_counter+=1; continue
                else:
                    n_correct+=1

            
            netG = Generator(img_size=img_size)
            netG = netG.to(device=device)
            optimizerG = optim.Adam(netG.parameters(), lr=args['gax_learning_rate'], betas=(0.9, 0.999))
            
            n_class = y.shape[1]
            co_score = -np.inf
            OPTIM_IMG_DIR = os.path.join(OPTIM_IMG_FOLDER,'op.%s.%s.%s'%(str(img_name),str(args['split']),str(submethod)))
            
            OPTIM_SCORE_DIR = os.path.join(OPTIM_IMG_FOLDER,'op.%s.%s.%s.COS'%(str(img_name),str(args['split']),str(submethod)))
            # print('\ntraining for method:GAX submethod:%s'%(str(submethod)))
            imgs, co_scores = [],[]

            if args['target_co']>0:
                args['n_iter'] = 0

            i=0            
            while True:                
                # train normally
                netG, co_score = one_step(netG,x,y, y_pred, n_class, submethod=args['submethod'])
                co_scores.append(co_score.item())

                if i==0 or (i+1)%4==0 or (i+1)==args['n_iter']:
                    main_update_str = '%s/%s oc:%.8f'%(str(i+1),str(args['n_iter']),co_score.item())
                    print('%-64s'%(str(main_update_str)),end='\r')
                    
                netG.eval()
                attrg = netG(x)
                attrg = attrg.clone().detach().cpu().numpy()[0].transpose(2,1,0)
                imgs.append(attrg)

                i+=1
                if args['target_co']==0:
                    if i>=args['n_iter']: break
                else:
                    if co_score.item()>args['target_co']:
                        print('img_name:%s target achieved at iter %s. co_score:%s submethod:%s'%(str(img_name),str(i),str(co_score.item()),str(submethod)))
                        break

            data = np.array(imgs)
            np.save(OPTIM_IMG_DIR, data)
            np.save(OPTIM_SCORE_DIR, np.array(co_scores))

            index_counter+=1
            if n_correct>=max_count: break
        return True

    def display_co_score_boxplot_for_existing_methods(self):
        args = self.args
        DIRS = self.DIRS

        DIRS = manage_xai_result_dir(DIRS, args)
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


    def display_co_score_for_existing_methods(self):
        args = self.args
        DIRS = self.DIRS 
        METHOD = args['method']

        DIRS = manage_xai_result_dir(DIRS, args)
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

    def collect_co_score_for_existing_methods(self):
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
        
        net = self.load_model(DIRS)
        DIRS = manage_xai_result_dir(DIRS,args)        
        XAI_COLLECT = load_or_create_result_dictionary(DIRS['XAI_COLLECT_DIR'])

        IMG_FOLDER = os.listdir(DIRS['IMG_FOLDER_DIR'])
        n_img = len(IMG_FOLDER)
        for i,img_name in enumerate(IMG_FOLDER):
            # ONE image per loop
            if args['n_debug']>0: 
                if i>=args['n_debug']: break 

            THIS_IMG_DIR = os.path.join(DIRS['IMG_FOLDER_DIR'], img_name)
            x, y0 = get_img_label_pair(IMG_DIR=THIS_IMG_DIR, LABEL=args['label'], img_size=img_size, device=device)

            attr = self.get_attribution_by_method(net, x, y0, **args)
            for submethod in ['sum','mult']:
                XAI_COLLECT, alrd_registered = self.compute_co_score_by_method(net, x, y0, attr, 
                    XAI_COLLECT, method=METHOD, submethod=submethod,
                    img_name=img_name, split=SPLIT)

                if (i+1)%4==0 or (i+1)==n_img:
                    update_str = '%s/%s | %s | submethod:%s | alrd_registered?:%s | n=%s'%(str(i+1),str(n_img), str(img_name),
                        str(submethod), str(alrd_registered),str(len(XAI_COLLECT)))
                    print('%-96s'%(str(update_str)),end='\r')

        print('')
        self.save_manager.pickle_data(XAI_COLLECT, DIRS['XAI_COLLECT_DIR'], tv=(0,0,100), text=None)

        
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

    def run_XAI_on_img(self):
        print('run_XAI_on_img')
        args = self.args
        DIRS = self.DIRS
        gax_split = args['split']
        gax_label = args['label']
        img_size = (args['img_size'],args['img_size']) 

        net = self.load_model(DIRS)
        print('loaded at iter %s'%(str(net.iter[0].item())))

        # gax_img_name = 'IM-0010-0001.jpeg' if args['img_name'] is None else args['img_name'] 
        # find the first correct prediction
        TEST_IMG_FOLDER_DIR = os.path.join(DIRS['DATA_DIR'], gax_split, gax_label)
        for gax_img_name in os.listdir(TEST_IMG_FOLDER_DIR):
            TEST_IMG_DIR = os.path.join(TEST_IMG_FOLDER_DIR ,gax_img_name) # , 'IM-0009-0001.jpeg' ) #
            x, y0 = get_img_label_pair(IMG_DIR=TEST_IMG_DIR, LABEL=gax_label, img_size=img_size, device=device)
            # print(x.shape,y0) # torch.Size([1, 3, 256, 256]) tensor([0], device='cuda:0')
            with torch.no_grad():
                y = net(x)
                y_pred = torch.argmax(y,dim=1)
                print('%s y_pred:%s, y0:%s'%(str(TEST_IMG_DIR),str(y_pred[0].item()),str(y0[0].item())))

            if y_pred[0].item()==y0[0].item():
                break

        output_name = '%s.%s.%s.%s.jpeg'%(str(args['method']),str(gax_split),str(gax_label),str(gax_img_name))
        SAVE_IMG_DIR = os.path.join(DIRS['SAVE_IMG_FOLDER'],output_name)
        attr = self.get_attribution_by_method(net, x, y0, **args)
        show_img_attr(x, y0, attr, output_name, SAVE_IMG_DIR=SAVE_IMG_DIR)

    def load_model(self, DIRS):
        args = self.args
        from .model import Resnet34Pneu
        model = torch.load(DIRS['MODEL_DIR'])
        if args['model'] == 'resnet34':
            from .model import Resnet34Pneu
            net = Resnet34Pneu()    
        elif args['model'] == 'alexnet':
            from .model import AlexPneu
            net = AlexPneu() 
        else:
            raise NotImplementedError()
        net = net.to(device=device)       
        net.load_state_dict(model['net'])
        net.eval()
        return net

    def display_gax_optimized_images(self):
        args = self.args
        DIRS = self.DIRS
        img_size = (args['img_size'],args['img_size']) 
        IMG_DIR = os.path.join(DIRS['DATA_DIR'], args['split'], args['label'], args['img_name'])
        OPTIM_IMG_FOLDER = os.path.join(DIRS['PROJECT_DIR'],'gax_images')
        OPTIM_IMG_DIR = os.path.join(OPTIM_IMG_FOLDER,'op.%s.%s.%s.npy'%(str(args['img_name']),str(args['split']),str(args['submethod'])))
        OPTIM_SCORE_DIR = os.path.join(OPTIM_IMG_FOLDER,'op.%s.%s.%s.COS.npy'%(str(args['img_name']),str(args['split']),str(args['submethod'])))
        
        LABEL, LABEL_INDEX = get_label(args)
        x, y0 = get_img_label_pair(IMG_DIR=IMG_DIR, LABEL=LABEL, img_size=img_size, device=device)
        
        data = np.load(OPTIM_IMG_DIR)
        co_scores = np.load(OPTIM_SCORE_DIR)
        D ,W, H, C = data.shape

        plt.figure()
        plt.imshow(x.clone().detach().cpu().numpy()[0].transpose(2,1,0))
        plt.gca().set_xlabel(LABEL)

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
            change=change, co_scores=co_scores, x=x, y0=y0)  
        slider_depth = ip.setup_slider(D)
        ip.update(0,init=True)
        slider_depth.on_changed(ip.update)
        plt.show()


def show_img_attr(img, y0, attribution, title=None, is_tensor=True, SAVE_IMG_DIR=None):
    if is_tensor:
        img = img[0].clone().detach().cpu().numpy().transpose(2,1,0)    
        h_raw = attribution[0].clone().detach().cpu().numpy().transpose(2,1,0)

    font = {'size': 7}
    plt.rc('font', **font)

    row, col = 2, 3
    plt.figure()
    plt.gcf().add_subplot(row,col,1)
    plt.gca().imshow(img)
    if not title is None: 
        plt.gca().set_title(title)

    def off_ticks():
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

    def overlay(pos,h, title=None):
        plt.gcf().add_subplot(row,col,pos)
        plt.gca().imshow(h, vmin=-1., vmax=1, cmap='bwr')
        plt.gca().imshow(img,alpha=0.1)
        off_ticks()
        if not title is None:
            plt.gca().set_title(title)
    
    h_sum = np.sum(h_raw, axis=2)
    h_sum_norm = np.max(np.abs(h_sum))
    overlay(2,h_sum/h_sum_norm, title='sum [%.3f]'%(h_sum_norm))

    h_abs_sum = np.sum(np.abs(h_raw), axis=2)
    h_abs_sum_norm = np.max(np.abs(h_abs_sum))
    overlay(3, h_abs_sum/h_abs_sum_norm, title='abs sum [%.3f]'%(h_abs_sum_norm))

    h_norm = np.max(np.abs(h_raw))
    h = h_raw/h_norm
    if h.shape[2]==3:
        colors = ['r','g','b']
        for i,pos in enumerate([4,5,6]):  
            overlay(pos, h[:,:,i], title='%s [%.3f/%.3f]'%(str(colors[i]),np.max(h[:,:,i]),h_norm))
    # plt.tight_layout()
    if SAVE_IMG_DIR is None:
        plt.show()
    else:
        plt.savefig(SAVE_IMG_DIR)


def get_img_label_pair(IMG_DIR, LABEL, img_size, for_pytorch=True, device=None):
    from .data import ImageProcessor
    loader = ImageProcessor(resize=img_size) # just borrow for img processing
    img = loader.simple_img_preprocessing(pil_img=Image.open(IMG_DIR))
    if LABEL == 'NORMAL':
        y0 = 0
    elif LABEL == 'PNEUMONIA':
        y0 = 1
    else:
        raise Exception('wrong label')

    if for_pytorch:
        x = np.array([img])
        y0 = np.array([y0])

        x = torch.from_numpy(x).to(torch.float)
        y0 = torch.from_numpy(y0).to(torch.long)

        if device is not None:
            x = x.to(device=device)
            y0 = y0.to(device=device)

    return x, y0

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

        self.img = kwargs['x'].clone().detach().cpu().numpy()[0].transpose(2,1,0)
        self.y0 = kwargs['y0'].clone().detach().cpu().numpy()[0]

    def update(self, val, init=False):
        # val is automatically slider_depth.val
        if init:
            current_depth = 0
        else:
            current_depth = int(val)

        x = self.data[current_depth,:,:,:]
        amax = np.max(np.abs(x))
        for i,name in enumerate(['',2,3]):
            getattr(self,'ax%s'%(str(name))).imshow(x[:,:,i], vmin=-1.,vmax=1., cmap='bwr')
            # getattr(self,'ax%s'%(str(name))).imshow(self.img,alpha=0.2)

        self.ax2.set_xlabel('abs max:%s'%(str(amax)))
        self.ax3.set_xlabel('step: %s score:%s'%(str(current_depth),
            str(np.round(self.kwargs['co_scores'][current_depth],3))))
        
        c = self.kwargs['change'][current_depth,:,:,:]
        self.ax4.imshow(c[:,:,0], cmap='Reds', vmin=0.)
        self.ax5.imshow(c[:,:,1], cmap='Reds', vmin=0.)
        self.ax6.imshow(c[:,:,2], cmap='Reds', vmin=0.)
                
        if not init: self.fig.canvas.draw_idle()
 
    def setup_slider(self,D):
        axcolor = 'lightgoldenrodyellow'
        posx, posy, widthx_fraction, widthy_fraction = 0.1, .93, 0.3, 0.05
        ax_depth = plt.axes([posx, posy, widthx_fraction, widthy_fraction], facecolor=axcolor)
        slider_depth = Slider(ax_depth, 'Time step', 0, D-1, valinit=0, valstep=1)
        return slider_depth 