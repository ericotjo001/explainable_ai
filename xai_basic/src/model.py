from .utils import *

# SPA: Spatial Product Attention
# SPA is roughly designed as a hybrid between convolutional block 
# and a transformer query, key system

class SPA(nn.Module):
    def __init__(self, input_c=3, out_c=48):
        super(SPA, self).__init__()
        self.cs = cs = {
            '0': 7,
            '1': 7,
            '2': 17,
            'out_c':out_c,
        } 
        self.act = nn.ModuleDict({ # if we reuse one ReLU for all processes, some Captum function will scream...
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
        # xqxk = torch.matmul(x[:,:self.c,:] , torch.transpose(x[:,self.c:, :],1,2))/(self.c**0.5)
        # print('xqxk.shape:',xqxk.shape) # (b,c,c)
        x = self.softmax(torch.matmul(x[:,:self.cs['out_c'],:] , 
            torch.transpose(x[:,self.cs['out_c']:, :],1,2))/(self.cs['out_c']**0.5))
        return x

class mabSPA(SPA):
    # magnitude-banded SPA
    def __init__(self, input_c=3, out_c=48, fc_output_c=3, forward_mode='label+heatmap'):
        super(mabSPA, self).__init__(input_c=input_c, out_c=out_c)
        self.fc = nn.Linear(out_c**2,fc_output_c)

        cs = self.cs
        self.mab = nn.ModuleDict({
            '0': nn.ConvTranspose2d(out_c, cs['2'], 7, stride=2,bias=True),
            '1': nn.ConvTranspose2d(cs['2'],cs['1'],3, stride=2,bias=True ),
            '2': nn.ConvTranspose2d(cs['1'],cs['0'],3, stride=2,bias=True ),
            '3': nn.ConvTranspose2d(cs['0'],3,3, stride=2,bias=False ), # 2*c for Q and K, like transformer
            })

        self.forward_mode = forward_mode
            
    def forward(self,x):
        for i in ['0','1','2','3']:
            x = self.act[i](self.bns[i](self.convs[i](x))) 

        if self.forward_mode == 'label+heatmap':
            # currently, x is xqxk with a query/key abstraction, like transformer
            # let's give xk a meaning by setting it corresponds to our heatmaps 
            h = x[:,self.cs['out_c']:, :].clone() # xq
            for i in ['0','1','2','3']:
                h = self.mab[i](h)

        b,cqk,_,_ = x.shape
        x = x.view(b,cqk,-1)
        # xqxk = torch.matmul(x[:,:self.cs['out_c'],:] , torch.transpose(x[:,self.cs['out_c']:, :],1,2))/(self.cs['out_c']**0.5)
        # print('xqxk.shape:',xqxk.shape) # (b,c,c)
        x = self.softmax(torch.matmul(x[:,:self.cs['out_c'],:] , 
            torch.transpose(x[:,self.cs['out_c']:, :],1,2))/(self.cs['out_c']**0.5))

        y = self.fc(x.view(b,-1))

        if self.forward_mode == 'label+heatmap':
            return y, h
        else: 
            return y

    def select_first_layer(self):
        return self.convs['0'] # just for GuidedGradCAM
