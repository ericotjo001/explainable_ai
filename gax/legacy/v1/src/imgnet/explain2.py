from .explain import *

def manage_xai_result2_dir(DIRS, args):
    SPLIT = args['split']    

    XAI_COLLECT_FOLDER = os.path.join(DIRS['PROJECT_DIR'], 'xai_collect2')
    if not os.path.exists(XAI_COLLECT_FOLDER): os.mkdir(XAI_COLLECT_FOLDER)
    XAI_COLLECT_DIR =  os.path.join(XAI_COLLECT_FOLDER, 'xai_collect2.dictionary')
    DIRS['XAI_COLLECT_FOLDER'] = XAI_COLLECT_FOLDER
    DIRS['XAI_COLLECT_DIR'] = XAI_COLLECT_DIR
    
    # IMG_FOLDER_DIR = os.path.join(DIRS['DATA_DIR'], SPLIT, args['label'])
    # DIRS['IMG_FOLDER_DIR'] = IMG_FOLDER_DIR
    return DIRS

class GAX2(GAX):
    def __init__(self, args):
        super(GAX2, self).__init__(args)
        self.args = args
        self.DIRS = manage_directories(args)

    def get_attribution_by_method(self, net, x, y0, **args):
        # override original method
        assert(x.shape[0]==1) # process ONE at a time
        x.requires_grad = True

        y_attr = net(x)
        y_attr = torch.argmax(y_attr,dim=1)

        method, layer = args['method'].split('.')

        if method =='LayerGradCam':
            img_size = (args['img_size'],args['img_size']) 
            if args['model'] == 'resnet34':
                attr = LayerGradCam(net, layer=getattr(net.backbone, layer)).attribute(x, target=y_attr)

                if layer in ['layer1','layer2','layer3','layer4']:
                    attr = torch.cat((attr,attr,attr),dim=1)
            else:
                raise NotImplementedError()
            
            attr = resize(attr[0].clone().detach().cpu().numpy().transpose(2,1,0), img_size + (3,))
            attr = torch.from_numpy(np.array([attr.transpose(2,1,0)])).to(device=device)
            # see shape in Saliency

        else:
            raise RuntimeError('Invalid method')
        
        return attr

    def collect_co_score_for_layer_comparison_and_display(self):
        self.collect_co_score_for_existing_methods(dir_method=manage_xai_result2_dir)
        self.display_co_score_for_layer_comparison()

    def display_co_score_for_layer_comparison(self):
        self.display_co_score_for_existing_methods(dir_method=manage_xai_result2_dir)

    def display_co_score_boxplot_for_layer_comparison(self):
        self.display_co_score_boxplot_for_existing_methods(dir_method= manage_xai_result2_dir)