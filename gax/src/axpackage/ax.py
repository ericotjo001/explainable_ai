from ..utils import *
from skimage.transform import resize

from captum.attr import Saliency , InputXGradient, LayerGradCam
from captum.attr import Deconvolution, GuidedBackprop, DeepLift

IMG_SIZE = (256,256)
softmax = nn.Softmax(dim=-1)

def compute_co_score(y_baseline, y, y0, n_class):
    # print(y_baseline.shape) # torch.Size([1, 1000])
    # print(y.shape) # torch.Size([1, 1000])
    # print(y0) # tensor([65])

    score_constants = torch.zeros_like(y) - 1./(n_class-1)
    score_constants[0,y0] = 1.
    co_score_components = (y - y_baseline)*score_constants
    # print(co_score_components.shape) # shape (b, c) where c is the no. of output channels

    co_score = torch.sum(co_score_components).item() # machine interpretability credit score    
    # print(co_score) # like 4.28281
    return co_score

class AugEplanation():
    def __init__(self, method='Saliency', model=None):
        super(AugEplanation, self).__init__()
        self.method = method
        self.model = model


    def get_attribution_by_method(self, net, x):
        assert(x.shape[0]==1) # process ONE at a time
        x.requires_grad = True
        y_attr = net(x)
        y_attr = torch.argmax(y_attr,dim=1)
        
        method = self.method
        model = self.model        
        if method=='Saliency':
            attr = Saliency(net).attribute(x, target=y_attr, abs=False)
            # print(attr.shape) # torch.Size([1, 3, 256, 256])
        elif method=='InputXGradient':
            attr = InputXGradient(net).attribute(x, target=y_attr)
        elif method =='LayerGradCam':
            if model in ['resnet34', 'resnet34_sub']:
                attr = LayerGradCam(net, layer=getattr(net.backbone,'conv1')).attribute(x, target=y_attr)
                # print(attr.shape) # torch.Size([1, 1, 128, 128])
                attr = resize(attr[0].clone().detach().cpu().numpy().transpose(1,2,0), IMG_SIZE + (3,))
                # print(attr.shape, x.shape) # (256, 256, 3) torch.Size([1, 3, 256, 256])

                # Note: in version 1, we transpose (2,1,0) and later transpose back with (2,1,0)
                #   We change it to (1,2,0), which is exactly the same so long as we transpose it back properly later

                # just to verify
                # x_img = x[0].clone().detach().cpu().numpy().transpose(1,2,0)
                # attr_img = attr
                # plt.figure()
                # plt.gcf().add_subplot(2,1,1)
                # plt.gca().imshow(attr_img/np.max(np.abs(attr_img)))
                # plt.gcf().add_subplot(2,1,2)
                # plt.gca().imshow(x_img) # remember the image is normalized by transform
                # plt.show()

                attr = torch.from_numpy(np.array([attr.transpose(2,0,1)])).to(device=device)

            elif model in ['alexnet']:
                attr = LayerGradCam(net, layer=net.backbone.features[0]).attribute(x, target=y_attr)
                attr = resize(attr[0].clone().detach().cpu().numpy().transpose(1,2,0), IMG_SIZE + (3,))
                attr = torch.from_numpy(np.array([attr.transpose(2,0,1)])).to(device=device)
            elif model in ['CXCMultiSPA']:
                attr = LayerGradCam(net, layer=net.spa1.convs['0']).attribute(x, target=y_attr)

                attr = resize(attr[0].clone().detach().cpu().numpy()[0], IMG_SIZE) 
                # print(attr.shape) # (256, 256)
                attr = torch.from_numpy(np.array([attr])).to(device=device)
                # print(attr.shape) # torch.Size([1, 256, 256])

            elif model in ['ccfFPA', 'drybeanFPA']:
                attr = LayerGradCam(net, layer=net.fpa.fconvs['0']).attribute(x, target=y_attr, attribute_to_layer_input=True)
                # print(attr.shape ,x.shape) # torch.Size([1, 1, 28]) torch.Size([1, 28])
                attr = attr.squeeze(1)
            else:
                raise NotImplementedError()
            
        elif method=='Deconvolution':
            attr = Deconvolution(net).attribute(x, target=y_attr)
        elif method=='GuidedBackprop':
            attr = GuidedBackprop(net).attribute(x, target=y_attr)
        elif method =='DeepLift':
            attr = DeepLift(net).attribute(x, target=y_attr)
        else:
            raise RuntimeError('Invalid method')
        
        return attr       

    def normalize_attr(self, attr):
        attr = attr.detach()
        attr_norm = torch.max(torch.abs(attr))
        attr = attr/attr_norm
        return attr        