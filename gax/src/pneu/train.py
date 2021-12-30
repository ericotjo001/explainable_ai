import os, time
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:',device)

from .data import PneuDataLoader

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.val_counter = 0
        self.args = args

    def get_loader(self, DIRS, split='train',n_debug=None, realtime_print=False):
        args = self.args
        start = time.time()
        img_size = (args['img_size'],args['img_size']) 
        loader = PneuDataLoader(DATA_DIR=DIRS['DATA_DIR'], split=split, 
            resize=img_size, n_debug=n_debug, realtime_print=realtime_print)
        print('data_size:',loader.data_size)
        end = time.time()
        elapsed = end - start
        print('[%s]loader time taken %s[s] = %s [min] = %s [hr]\n'%(str(split),str(round(elapsed,1)), 
            str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))))
        return loader
        
    def train(self,):
        print('train()')

        args = self.args
        batch_size = 8 if args['show_batch'] else args['batch_size']

        from .utils import manage_directories
        DIRS = manage_directories(args)

        if args['model'] == 'resnet34':
            from .model import Resnet34Pneu
            net = Resnet34Pneu()    
        elif args['model'] == 'alexnet':
            from .model import AlexPneu
            net = AlexPneu() 
        else:
            raise NotImplementedError()
        net = net.to(device=device)
        optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5,0.999), weight_decay=1e-5)
        if os.path.exists(DIRS['MODEL_DIR']):
            model = torch.load(DIRS['MODEL_DIR'])
            net.load_state_dict(model['net'])
            optimizer.load_state_dict(model['optimizer'])
            print('model loaded at iter:%s'%(str(net.iter[0].item())))
                
        criterion = nn.CrossEntropyLoss()

        #  [train]loader time taken 1004.8[s] = 16.7 [min] = 0.3 [hr]
        trainloader = self.get_loader(DIRS, split='train', n_debug=args['n_debug'], realtime_print=args['realtime_print'])
        valloader = self.get_loader(DIRS, split='val', n_debug=args['n_debug'], realtime_print=args['realtime_print'])

        print('\nStart training...')
        start = time.time()
        for i in range(args['n_iter']):
            net.train()
            net.zero_grad()

            x,y0 = trainloader.get_data(batch_size=batch_size, device=device)
            if args['show_batch']: show_batch_of_eight(x,y0)

            y = net(x)

            loss = criterion(y,y0)
            loss.backward()
            optimizer.step()
            net.iter[0] = net.iter[0]+1

            if args['realtime_print']:
                if (i+1)%8==0 or (i+1)==args['n_iter']:
                    y_pred = torch.argmax(y, dim=1).clone().detach().cpu().numpy()
                    update_str = '%s/%s y_pred:%s y0:%s loss:%s'%(str(i+1),str(args['n_iter']),
                        str(y_pred),str(y0.clone().detach().cpu().numpy()),str(loss.item()))
                    print('%-64s'%(str(update_str),),end='\r')


            if ((i+1)%100==0 or (i+1)==args['n_iter']) and i>args['min_iter']:
                EARLY_STOP = self.do_validate(valloader, net, VALIDATION_ACC=args['VALIDATION_ACC'])
                if EARLY_STOP: 
                    print('\nValidation acc achieved, early stopping.\n');break

        end = time.time()
        elapsed = end - start
        print('\ntraining time taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))))

        model = { 'net': net.state_dict(),'optimizer': optimizer.state_dict(),}
        torch.save(model, DIRS['MODEL_DIR'])
        print('model saved at iter:%s'%(str(net.iter[0].item())))

    def do_validate(self, valloader, net, VALIDATION_ACC=0.8):
        net.eval()

        val_counter = 0
        with torch.no_grad():
            for i in range(valloader.data_size):
                x,y0 = valloader.get_data(index=i,mode='by_index',device=device)
                y = net(x)
                y_pred = torch.argmax(y, dim=1)[0].item()
                # print(i,y_pred, y0[0].item(), y, y.shape)

                if int(y_pred)==int(y0[0].item()):
                    val_counter += 1
        acc = val_counter/valloader.data_size
        
        EARLY_STOP = True if acc>VALIDATION_ACC else False 
        if EARLY_STOP:
            print('do_validate(), EARLY_STOP activated. valloader.data_size:%s, acc:%s'%(str(valloader.data_size),str(acc)))
        return EARLY_STOP

    def evaluate(self,):
        args = self.args
        
        from .utils import manage_directories
        DIRS = manage_directories(args)        

        testloader = self.get_loader(DIRS, split='test', n_debug=args['n_debug'])

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

        TP, TN, FP, FN = 0,0,0,0
        with torch.no_grad():
            for i in range(testloader.data_size):

                x,y0 = testloader.get_data(index=i,mode='by_index',device=device)
                y = net(x)
                y_pred = torch.argmax(y, dim=1)[0].item()         
                
                if int(y_pred)==int(y0[0].item()):
                    if int(y0[0].item()) == 0:
                        TN+=1
                    else:
                        TP+=1
                else:
                    if int(y0[0].item()) == 0:
                        FP += 1
                    else:
                        FN += 1

                if args['realtime_print']:
                    if (i+1)%4==0 or (i+1)==testloader.data_size:
                        update_str = '%s/%s'%(str(i+1),str(testloader.data_size))
                        print('%-64s'%(str(update_str)),end='\r')
        acc = (TP+TN)/testloader.data_size
        recall = TP/(TP+FN) 
        precision = TP/(TP+FP)
        result_dict = {'n_tested': testloader.data_size,'acc': acc,
            'recall':recall,'precision':precision}
        print('\n', result_dict)
        
        import json
        with open(DIRS['RESULT_DIR'], 'w') as json_file:
            json.dump(result_dict, json_file, indent=4, sort_keys=True)



import matplotlib.pyplot as plt
def show_batch_of_eight(x, y0):
    # x is a tensor of size (8,C,H,W)
    # print(x.shape) # torch.Size([8, 1, 960, 960])
    # print(y0) # tensor([0, 0, 1, 0, 1, 0, 0, 1], dtype=torch.int32)

    plt.figure(figsize=(8,5))
    for i in range(8):
        plt.gcf().add_subplot(2,4,i+1)
        plt.gca().imshow(x[i].clone().detach().cpu().numpy().transpose(2,1,0), cmap='gray')
        plt.gca().set_title(y0[i].item())
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
    plt.tight_layout()
    plt.show()
    exit()


