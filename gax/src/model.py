from .utils import *
import torchvision.models as mod

"""
All the adjustments made in this script are necessary
because pytorch captum may throw some errors otherwise.
For example, Captum doesn't like activation function modules to be reused.
"""

def pretrained_model_selection_for_imagenet(model):
    if model == 'resnet34':
        net = ResNet34Adj().to(device=device)
    elif model == 'alexnet':
        net = AlexAdj().to(device=device)
    else:
        raise NotImplementedError()
    return net

def pretrained_model_selection_for_chestxray_pneu(model, pretrained_folder_dir):
    if model in ['resnet34', 'resnet34_sub']:
        net = Resnet34Pneu().to(device=device)
    elif model in ['alexnet']:
        net = AlexPneu().to(device=device)
    else:
        raise NotImplementedError()

    MODEL_DIR = os.path.join(pretrained_folder_dir, f"pretrained_chestxray_{model}.model")
    
    ckpt = torch.load(MODEL_DIR)
    net.load_state_dict(ckpt['net'])
    net = net.to(device=device)
    return net    

class BasicBlockAdjusted(mod.resnet.BasicBlock):
    def __init__(self):
        super(BasicBlockAdjusted, self).__init__(3, 3,) # dummy inputs
        # we will use inherit_weights() to get all its components
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

    def inherit_weights(self, BasicBlock):
        self.conv1 = BasicBlock.conv1
        self.bn1 = BasicBlock.bn1
        self.conv2 = BasicBlock.conv2
        self.bn2 = BasicBlock.bn2
        self.downsample = BasicBlock.downsample
        self.stride = BasicBlock.stride
        


class ResNet34Adj(nn.Module):
    def __init__(self):
        super(ResNet34Adj, self).__init__()

        from torchvision.models import resnet34, ResNet34_Weights
        self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device=device) 
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
        from torchvision.models import alexnet, AlexNet_Weights
        self.backbone = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(device=device)
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


class Resnet34Pneu(nn.Module):
    def __init__(self):
        super(Resnet34Pneu, self).__init__()
        self.iter = nn.Parameter(torch.from_numpy(np.array([0.])), requires_grad=False)
    
        from torchvision.models import resnet34, ResNet34_Weights
        self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device=device) 
        self.fc = nn.Linear(512, 2, bias=False)

        self.adjust_for_captum_problem()
        # self.print_after_captum_adjustment() # or debugging

    def forward(self,x):
        for i,m in enumerate(self.backbone.children()):
            if i>8: break
            x = m(x)
        # print(x.shape) # torch.Size([4, 512, 1,1])
        x = x.squeeze(3).squeeze(2)
        x = self.fc(x)
        # print(x.shape) # torch.Size([4, 2])
        return x

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

    def print_after_captum_adjustment(self):
        for i,(layer_name,m) in enumerate(self.backbone.named_children()):
            print(type(m))
            for j,(sublayer_name,m2) in enumerate(m.named_children()):
                print(type(m2))
                for k,(subsublayer_name,m3) in enumerate(m2.named_children()):
                    print(type(m3))

class AlexPneu(nn.Module):
    def __init__(self,):
        super(AlexPneu, self).__init__()
        self.iter = nn.Parameter(torch.from_numpy(np.array([0.])), requires_grad=False)
    
        from torchvision.models import alexnet, AlexNet_Weights
        self.backbone = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(device=device)
        self.fc= nn.Linear(4096,2,bias=False)    
        self.adjust_for_captum()

    def forward(self,x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        for i,m in enumerate(self.backbone.classifier.children()):
            if i==6: break # skip last layer with 1000 classes, change it with your own linear layer
            x = m(x)
        x = self.fc(x)
        return x   

    def adjust_for_captum(self):
        # deal with all the inplace relu problems
        for i,m in enumerate(self.backbone.features.children()):
            if type(m)==nn.ReLU:
                # print('yes', type(self.backbone.features[i]))
                self.backbone.features[i] = nn.ReLU() # without inplace=True
        for i,m in enumerate(self.backbone.classifier.children()):
            if type(m)==nn.ReLU:
                self.backbone.classifier[i] = nn.ReLU() # without inplace=True




# Spa-based deep neural network for classifying covid cases using chest x-ray images
# It's first introduced in paper "Quantifying Explainability of Saliency Methods in Deep Neural Networks with a Synthetic Dataset"
# The main concepts are 1. to quickly downsample the images 2. use query-key abstraction, like a transformer

class CXCMultiSPA(nn.Module):
    """Chest X-Ray COVID MultiSPA"""
    def __init__(self, fc_dim=16*16):
        super(CXCMultiSPA, self).__init__()
        self.spa1 = SPA(input_c=1, out_c=128)
        self.decon = DeconvSeq()

        self.fc_dim = fc_dim
        self.spa2 = SPA(input_c=3, out_c=int(fc_dim**0.5))
        self.fc = nn.Linear(fc_dim,4)
        
    def forward(self,x):
        x = self.spa1(x).unsqueeze(1)
        x = self.decon(x)
        x = self.spa2(x)
        x = self.fc(x.reshape(-1, self.fc_dim))
        return x


class SPA(nn.Module):
    def __init__(self, input_c=3, out_c=48):
        super(SPA, self).__init__()
        self.cs = cs = {
            '0': 7,
            '1': 7,
            '2': 17,
            'out_c':out_c,
        } 

        # if we reuse one ReLU for all processes, some Captum function will scream...
        self.act = nn.ModuleDict({ 
            '0': nn.ReLU(),
            '1': nn.ReLU(),
            '2': nn.ReLU(),
            '3': nn.ReLU(),            
        })
        self.softmax = nn.Softmax(dim=-1)

        self.convs = nn.ModuleDict({
            '0': nn.Conv2d(input_c, cs['0'], 7, stride=2,),
            '1': nn.Conv2d(cs['0'], cs['1'], 3, stride=2),
            '2': nn.Conv2d(cs['1'], cs['2'], 3, stride=2),
            '3': nn.Conv2d(cs['2'], 2*cs['out_c'] , 3, stride=2), # 2*c for Q and K, like transformer
            })
        self.bns = nn.ModuleDict({
            '0': nn.BatchNorm2d(cs['0']),
            '1': nn.BatchNorm2d(cs['1']),
            '2': nn.BatchNorm2d(cs['2']),
            '3': nn.BatchNorm2d(2*cs['out_c']),
            })
        
    def forward(self,x):
        for i in ['0','1','2','3']:
            x = self.act[i](self.bns[i](self.convs[i](x)))

        b,cqk,_,_ = x.shape
        x = x.view(b,cqk,-1)
        # xqxk = torch.matmul(x[:,:self.cs['out_c'],:] , 
        #       torch.transpose(x[:,self.cs['out_c']:, :],1,2))/(self.cs['out_c']**0.5)
        # print('xqxk.shape:',xqxk.shape) # (b,c,c)
        x = self.softmax(torch.matmul(x[:,:self.cs['out_c'],:] , 
            torch.transpose(x[:,self.cs['out_c']:, :],1,2))/(self.cs['out_c']**0.5))
        return x


class DeconvSeq(nn.Module):
    """Sequence of deconv"""
    def __init__(self, ):
        super(DeconvSeq, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(1,3,3,stride=2, bias=False)
        self.act = nn.LeakyReLU() 

    def forward(self,x):
        x = self.deconv1(x)
        x = self.act(x)
        return x       


class FPA(nn.Module):
    """
    Feature Product Attention
    Something like SPA, but for 1D
    We want query-key concept to be applied to 1D input too, just like SPA
    Since the no. of features aren't that large compared 2D pixels, we don't
    really need to use stride=2
    """
    def __init__(self, input_c=1, output_c=7):
        super(FPA, self).__init__()
        self.cs = cs = {
            '0': 7,
            'out_c': output_c,
        }

        # if we reuse one ReLU for all processes, some Captum function will scream...
        self.act = nn.ModuleDict({ 
            '0': nn.ReLU(),
            '1': nn.ReLU(),
            })

        self.softmax = nn.Softmax(dim=-1)
        self.fconvs = nn.ModuleDict({
            '0': nn.Conv1d(input_c, cs['0'], 7), 
            '1': nn.Conv1d(cs['0'], 2*cs['out_c'], 3),
            })
        
        self.relu = nn.ReLU()
        self.bns = nn.ModuleDict({
            '0': nn.BatchNorm1d(cs['0']),
            '1': nn.BatchNorm1d(2*cs['out_c']),
            })

        # self.fc2 = nn.Linear(2*input_c, 2*output_c)

    def forward(self, x):
        # x is assumed to be (b,d) i.e. typical one-d vector
        x = x.unsqueeze(1) # (b,1,d)

        for i in ['0','1']:
            x = self.act[i](self.bns[i](self.fconvs[i](x)))
        
        # b, cqk, c = x.shape
        x = self.softmax(torch.matmul(x[:,:self.cs['out_c'],:] , 
            torch.transpose(x[:,self.cs['out_c']:, :],1,2))/(self.cs['out_c']**0.5))

        # print(x.shape) # (b,output_c,output_c)
        return x

class ccfFPA(nn.Module):
    def __init__(self, ):
        super(ccfFPA, self).__init__()
        output_c = 7
        self.fpa = FPA(input_c=1, output_c=output_c)
        self.fc = nn.Linear(output_c**2,2)

    def forward(self,x):
        x = self.fpa(x)
        b,_,_ = x.shape
        x = self.fc(x.reshape(b,-1))
        return x
        
class drybeanFPA(nn.Module):
    def __init__(self):
        super(drybeanFPA, self).__init__()
        output_c = 14
        self.fpa = FPA(input_c=1, output_c=output_c)        
        self.fc = nn.Linear(output_c**2,7)

    def forward(self,x):
        x = self.fpa(x)
        b,_,_ = x.shape
        x = self.fc(x.reshape(b,-1))
        return x
        