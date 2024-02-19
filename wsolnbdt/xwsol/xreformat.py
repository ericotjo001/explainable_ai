import numpy as np
from skimage.transform import resize
import torch
import torch.nn as nn

def tonumpy(x):
    return x.clone().detach().cpu().numpy()



def heatmap_reformat(hmap, h, w, mode=None, layer=None):
    # hmap shape is (1,c,h1,w1)
    # we reformat to numpy and resize 
    b,c,h1,w1 = hmap.shape 
    if mode is None:
        out = hmap[0,0]
        out = resize(tonumpy(out), (h,w))
        out = out - np.min(out)
        out = divide_by_norm(out)   

    elif mode == 'unformatted':
        out = torch.sum(hmap ,1) 
        out = tonumpy(out)[0] 

        out = divide_by_norm(out)        
        out = resize(out, (h,w))              

    elif mode in ['saliency',]:
        """
        Main paper on saliency takes max over channel
        """
        out = torch.max(hmap,1).values 
        out = tonumpy(out)[0] 

        out = np.abs(out) # the main paper that introduces Saliency method does this
        out = divide_by_norm(out)        
        out = resize(out, (h,w), )

    elif mode in ['saliency_alt']:
        # the weights and pooling create a nice "dense" heatmap!
        weights = torch.sum(hmap.reshape(b,c,-1),2)
        out = torch.mean(hmap * weights.unsqueeze(-1).unsqueeze(-1) ,1) 
        out = nn.functional.avg_pool2d(out,2,stride=1,padding=1) 
        out = tonumpy(out)[0] 

        out = out - np.min(out)
        out = divide_by_norm(out)        
        out = resize(out, (h,w))

    elif mode=='gradcam':
        # print(hmap.shape) # in LayerGradCAM, sum over channel 2 occurs, so we get (b,1,h,w)
        out = hmap[0,0]
        out = resize(tonumpy(out), (h,w))
        out = out - np.min(out)
        out = divide_by_norm(out)      

    elif mode in ['gbp',]:
        if layer in ['layer1', 'layer2','layer3',]:
            out = 1. - hmap # yes! this will make some layer perform localization
            out = torch.sum(out ,1, keepdims=True) 
            out = nn.functional.max_pool2d(out,2,stride=1,padding=1).squeeze(1)
        elif layer in ['layer4']:
            out = torch.sum(hmap ,1,) 
        else:
            out = torch.sum(hmap ,1,) 
            
        out = tonumpy(out)[0] 

        out = out - np.min(out)
        out = divide_by_norm(out)        
        out = resize(out, (h,w))      
    elif mode in ['deeplift','deeplift_alt']:
        out = torch.sum(hmap ,1) 
        out = tonumpy(out)[0] 

        out = out - np.min(out)
        out = divide_by_norm(out)        
        out = resize(out, (h,w))      
    else:
        raise NotImplementedError()
        
    return out

def divide_by_norm(x):
    div = np.max(np.abs(x)) 
    div = 1 if div==0 else div
    x = x/ div
    return x

def showcase(out):
    print(out.shape)
    plt.figure()
    plt.imshow(out/np.max(np.abs(out)), vmin=-0,vmax=1.)
    plt.colorbar()
    plt.show()
    exit()