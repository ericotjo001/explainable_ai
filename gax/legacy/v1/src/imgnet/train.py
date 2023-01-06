from .utils import manage_directories

import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torchvision.models as mod
import torchvision.transforms as transforms

from .imagenet_utils import ImageNetDirectQuery
from .adjusted_model_component import BasicBlockAdjusted

class ResNet34Adj(nn.Module):
    def __init__(self):
        super(ResNet34Adj, self).__init__()
        self.backbone = mod.resnet34(pretrained=True, progress=False).to(device=device)
        self.adjust_for_captum_problem()

    def adjust_for_captum_problem(self):
        setattr(self.backbone,'relu',nn.ReLU() )
        for i,(layer_name,m) in enumerate(self.backbone.named_children()):    
            if type(m) == nn.Sequential:
                for j,(sublayer_name,m2) in enumerate(m.named_children()):
                    if type(m2) == mod.resnet.BasicBlock:
                        # setattr(getattr(getattr(self.backbone, layer_name), sublayer_name),'relu', nn.ReLU())
                        temp = BasicBlockAdjusted()
                        temp.inherit_weights(m2)
                        setattr(getattr(self.backbone, layer_name), sublayer_name, temp)

    def forward(self, x):
        return self.backbone(x)

class AlexAdj(nn.Module):
    def __init__(self,):
        super(AlexAdj, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
        self.adjust_for_captum()

    def forward(self,x):
        return self.backbone(x)

    def adjust_for_captum(self):
        # deal with all the inplace relu problems
        for i,m in enumerate(self.backbone.features.children()):
            if type(m)==nn.ReLU:
                # print('yes', type(self.backbone.features[i]))
                self.backbone.features[i] = nn.ReLU() # without inplace=True
        for i,m in enumerate(self.backbone.classifier.children()):
            if type(m)==nn.ReLU:
                self.backbone.classifier[i] = nn.ReLU() # without inplace=True

def pretrained_model_selection(model_name, eval=True,device=None):
    if model_name == 'resnet34':
        print('getting pretrained model resnet34...')
        net = ResNet34Adj().to(device=device)
    elif model_name =='alexnet':
        print('getting pretrained model alexnet...')
        net = AlexAdj().to(device=device)
    else:
        raise RuntimeError('Invalid model selection.')
    
    net.to(device=device)
    if eval: net.eval()
    return net

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
    
        self.args = args
        self.DIRS = manage_directories(args)

    def train(self, ):
        print('train')
        args = self.args
        DIRS = self.DIRS

        # for fetching data
        imgn = ImageNetDirectQuery(MAIN_DATA_DIR=args['DATA_DIR'], N_DEBUG=args['n_debug_imagenet'])

        DESC = """This is just a training demo. We will not go too deep into this.\nThe model is pretrained and our main purpose is to apply the GAX method on it."""
        print(DESC)

        net = pretrained_model_selection(model_name=args['model'], eval=False, device=device)
        optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5,0.999))

        normalize = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])

        criterion = nn.CrossEntropyLoss()
        for i in range(args['n_iter']):
            net.zero_grad()
            x, y0 = imgn.get_pytorch_batch_samples(args['batch_size'], input_size=(256,256), device=device)
            # print(x.shape,y0) # torch.Size([4, 3, 256, 256]) tensor([0, 0, 0, 0], device='cuda:0')

            y = net(x)
            loss = criterion(y,y0)
            
            loss.backward()
            optimizer.step()
            # print(loss.item())
            if (i+1)%10==0 or (i+1)==args['n_iter']:
                update_str = '%s/%s'%(str(i+1),str(args['n_iter']))
                print('%-64s'%(str(update_str)),end='\r')

        print("\ntraining complete. Not saved, since we want to test the pretrained model's co_score performance.")

    def eval_selected_image(self):
        print('eval_selected_image')
        args = self.args
        DIRS = self.DIRS


        net = pretrained_model_selection(model_name=args['model'], eval=True, device=device)
        normalize = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])        


        import os, copy
        from PIL import Image
        import numpy as np
        import matplotlib.pyplot as plt
        SELECTED_IMG_DIR = os.path.join(DIRS['ROOT_DIR'], 'data', 'imgnet_selected', args['selected_img_name'])
        pil_img = Image.open(SELECTED_IMG_DIR)
        img = np.asarray(pil_img)/255.
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        img = img.transpose(2,0,1)
        # print(np.max(img), np.min(img))

        img = torch.from_numpy(np.array(img)).to(device=device).to(torch.float)
        img = img.unsqueeze(0)
        img = normalize(img)
        print(img.shape)
        y = net(img)
        y_pred = torch.argmax(y[0])

        from .imagenet_dict import LABEL_DIR
        print(y_pred, LABEL_DIR[str(y_pred.item())]['label'])