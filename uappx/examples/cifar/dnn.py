import os, time, joblib
import numpy as np
import torch
import torch.nn as nn
from src.utils import parse_bool_from_string, strbool_description, readjust_bools
import torchvision.models as mod
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalizeTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
normalizeImageTransform = transforms.Compose([transforms.ToTensor(), 
    normalizeTransform])

class ResNet18x(nn.Module):
    def __init__(self,):
        super(ResNet18x, self).__init__()
        self.backbone = mod.resnet18(pretrained=True, progress=False)
        self.fc = nn.Linear(512,10,)

    def forward(self,x, ):
        for i,m in enumerate(self.backbone.children()):
            if i>8: break
            x = m(x)
        x = self.fc(x.squeeze(-1).squeeze(-1))
        return x

def dnn_pipeline(parser, DIRS):
    parser.add_argument('--epoch', default=4, type=int, help=None)
    parser.add_argument('--batch_size', default=16, type=int, help=None)
    BOOLS = {
        'load': 1,
    }  
    for bkey,b in BOOLS.items():
        parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)

    train_dnn(dargs, DIRS)
    eval_dnn(dargs,DIRS, data_mode='train')
    eval_dnn(dargs, DIRS)

def train_dnn(dargs, DIRS):
    from .cifar_data import prepare_cifarloader
    # trainloader = prepare_dataloader(DIRS['SOURCE_DATA_DIR'], train=True, batch_size=dargs['batch_size'], shuffle=True)
    trainloader = prepare_cifarloader(train=True, root_dir=DIRS['SOURCE_DATA_DIR'], 
        batch_size=dargs['batch_size'], shuffle=True, demo=False, download=False)

    net = ResNet18x ()
    net.to(device=device)

    import torch.optim as optim
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5,0.999))
    
    criterion = nn.CrossEntropyLoss()
    n_train = len(trainloader)
    for epoch in range(dargs['epoch']):
        for i,data in enumerate(trainloader):
            net.zero_grad()
            x,y0 = data
            x,y0 = x.to(device=device).to(torch.float), y0.to(device=device).to(torch.long)

            y = net( normalizeTransform(x) )
            loss = criterion(y,y0)
            loss.backward()
            optimizer.step()

            if (i+1)%24==0 or (i+1) ==n_train:
                update_text = 'epoch:%s %s/%s'%(str(epoch+1),str(i+1),str(n_train))
                print('%-36s'%(str(update_text)), end='\r')
    print('\nTraining done!')
    torch.save(net, DIRS['CNN_MODEL_DIR'])


def eval_dnn(dargs, DIRS, data_mode='test'):
    from .cifar_data import prepare_cifarloader

    if data_mode=='test':
        train = False
        datatext = '<test data>'
    elif data_mode == 'train':
        train = True
        datatext = '<train data>'
    else:
        raise Exception('what?')

    testloader = prepare_cifarloader(train=train, root_dir=DIRS['SOURCE_DATA_DIR'], 
        batch_size=dargs['batch_size'], shuffle=False, demo=False, download=False)

    net = torch.load(DIRS['CNN_MODEL_DIR'])
    net.eval()

    print('evaluation starts!')
    n_test = 0
    n_correct = 0
    n_iter = len(testloader)
    for i,data in enumerate(testloader):
        x,y0 = data
        x,y0 = x.to(device=device).to(torch.float), y0.to(device=device).to(torch.long)
        y = net( normalizeTransform(x) )

        y_pred = torch.argmax(y, dim=1)
        for y1,y2 in zip(y_pred.clone().detach().cpu().numpy(),y0):
            if y1==y2.item():
                n_correct+=1
        n_test+= len(y0)        
        if (i+1)%240==0 or (i+1) ==n_test:
            update_text = '%s/%s'%(str(i+1),str(n_iter))
            print('%-36s'%(str(update_text)), end='\r')

    report_text = '%s\nacc:%s/%s=%s\n'%(str(datatext),str(n_correct),str(n_test),str(n_correct/n_test))
    print('\n',report_text)
    with open(DIRS['CNN_REPORT_DIR'] , 'a') as f:
        f.write(report_text)

