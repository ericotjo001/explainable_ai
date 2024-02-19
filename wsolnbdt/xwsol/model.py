import torch
import torch.nn as nn
import torchvision.models as mod
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize

try:
    from xwsol.adjusted_model_component import BottleneckAdjusted
except:
    pass

class ResNet50CAM(nn.Module):
    def __init__(self,):
        super(ResNet50CAM, self).__init__()
        try:
            self.backbone = mod.resnet50(pretrained=True, progress=False)
        except:
            print('LOADING MODEL FOR OLDER PYTORCH VERSION!')
            self.backbone = mod.resnet50(pretrained=True, )

        self.adjust_for_captum_problem()
        

    def forward(self,x, ):
        return self.backbone(x)

    def verify_equality(self,x):
        print('verify_equality...')
        with torch.no_grad():
            y1 = x.clone().detach()
            
            for mkey,_ in self.backbone.named_children():
                if mkey=='fc':
                    y1 = y1.squeeze(3).squeeze(2)
                y1 = self.backbone.__getattr__(mkey)(y1)

            y = self.backbone(x)
        print(y[0,:4])
        print(y1[0,:4])
        assert(np.all(tonumpy(y)==tonumpy(y1)))
        print('equality ok!')
        

    def compute_cam(self,x, labels):
        # inside self.backbone (mod.resnet50):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        with torch.no_grad():
            for mkey,_ in self.backbone.named_children():
                x = self.backbone.__getattr__(mkey)(x)
                if mkey =='layer4':
                    break

        feature_map = x.detach().clone()
        cam_weights = self.backbone.__getattr__('fc').weight[labels]
        cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                feature_map).mean(1, keepdim=False)
        return cams

    def adjust_for_captum_problem(self):
        """
        conv1
        bn1
        relu
        maxpool
        layer1
        layer2
        layer3
        layer4
        avgpool
        fc
        """
        setattr(self.backbone,'relu',nn.ReLU() )
        for i,(layer_name,m) in enumerate(self.backbone.named_children()):    
            if type(m) == nn.Sequential:
                for j,(sublayer_name,m2) in enumerate(m.named_children()):
                    if type(m2) == mod.resnet.Bottleneck:
                        # setattr(getattr(getattr(self.backbone, layer_name), sublayer_name),'relu', nn.ReLU())
                        temp = BottleneckAdjusted()
                        temp.inherit_weights(m2)
                        setattr(getattr(self.backbone, layer_name), sublayer_name, temp)

    #####################################
    # special construction for non-layer based captum modules
    # unlike DeepLift, Saliency has no option to output heatmap for other layer, e.g. layer 4
    #   Saliency always outputs the heatmap for input.
    # We create this to allow heatmap for specifically layer 4 output
    #####################################
    def create_split_at_layer(self, split_layer_name):
        self.mod_front = ModuleDictX()
        self.mod_back = ModuleDictX()
        front=True
        for i,(layer_name,m) in enumerate(self.backbone.named_children()):    
            if front:
                # print('front',layer_name)
                self.mod_front[layer_name] = m
            else:
                # print('back ',layer_name)
                self.mod_back[layer_name] = m        
                        
            if layer_name == split_layer_name:
                front = False


    def verify_split_equality(self,x):
        print('verify_split_equality...')
        with torch.no_grad():
            y1 = x.clone().detach()
            
            y1 = self.mod_front(y1)
            y1 = self.mod_back(y1)

            y = self.backbone(x)
        print(y[0,:4])
        print(y1[0,:4])
        assert(np.all(tonumpy(y)==tonumpy(y1)))
        print('equality ok!')

class ModuleDictX(nn.ModuleDict):
    """ Just a module to implement the forward function of ModuleDict
    """
    def __init__(self, ):
        super(ModuleDictX, self).__init__()
    
    def forward(self,x):
        for mkey,_ in self.items():
            if mkey=='fc':
                x = x.squeeze(3).squeeze(2)
            x = self[mkey](x)
        return x            
        
def tonumpy(x):
    return x.clone().detach().cpu().numpy()

if __name__ == '__main__':
    print('Pretrained ResNet50 with CAM formula')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import os
    try:
        from adjusted_model_component import BottleneckAdjusted
    except:
        pass

    img_dir = 'cat_dog.png' 
    img_dir = img_dir if os.path.exists(img_dir) else os.path.join('xwsol',img_dir) 
    
    pil_img = Image.open(img_dir)
    h,w,c = np.asarray(pil_img).shape
    img = torch.from_numpy(np.asarray(pil_img).transpose(2,0,1)).unsqueeze(0).to(device=device).to(torch.float)

    normalizeTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalizeImageTransform = transforms.Compose([transforms.ToTensor(), normalizeTransform])

    batch_size = 3
    x = torch.randn(size=(batch_size,3,h,w)).to(device=device).to(torch.float)
    x[0,:,:,:] = x[0,:,:,:]*0 + img
    x[1,:,:,:] = x[1,:,:,:]*0 + img
    labels = torch.randint(0,1000, size=(batch_size,)).to(device=device).to(torch.long)
    labels[0] = 243 # correct label for dog
    labels[1] = 281 # tabby cat
    print(x.shape)

    net = ResNet50CAM().to(device=device)
    net.verify_equality(x) # check that everything is correct first.
    net.verify_split_equality(x)

    cams = net.compute_cam(x, labels)
    print(cams.shape)
    # y = net(x,verbose=100)
    # print(y.shape)

    plt.figure()
    plt.gcf().add_subplot(131)
    plt.gca().imshow(pil_img)
    plt.gca().imshow(resize(tonumpy(cams[0,:,:]), (h,w)), alpha=0.5)
    plt.gcf().add_subplot(132)
    plt.gca().imshow(pil_img)
    plt.gca().imshow(resize(tonumpy(cams[1,:,:]), (h,w)), alpha=0.5)
    plt.gcf().add_subplot(133)
    plt.gca().imshow(tonumpy(x[2,:,:,:]).transpose(1,2,0))
    plt.gca().imshow(resize(tonumpy(cams[2,:,:]), (h,w)), alpha=0.5)
    plt.show()
    plt.close()