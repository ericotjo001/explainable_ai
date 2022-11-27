"""
mabcam is an entire pipeline to train model with the generation of CAM-like heatmaps
with a specific feature: have banded (roughly multi-modal) weights that 
correspond to neutral, localization and features. 
"""
from .utils import *
from .data import load_dataset_from_a_shard
from .model import mabSPA
import torch.optim as optim
from .objgen.random_simple_gen_implemented2 import ThreeClassesPyIOwithHeatmap 
from .objgen.random_simple_gen_implemented import TenClassesPyIOwithHeatmap 

DEBUG_TOGGLES = {
    'train_iter' : 0,
    'val_iter': 0,
}

_TARGET_ACC_ = 0.975
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_val_mabcam(dargs):
    # training and validation
    print('train_mabcam')

    DIRS = manage_dir(dargs)
    if dargs['n_classes'] == 3:
        net = mabSPA().to(device=device)
    elif dargs['n_classes'] == 10:
        net = mabSPA(fc_output_c=10).to(device=device)
    else:
        raise NotImplementedError()
    print('n params:', count_parameters(net))

    # the problem is simple enough not to need extreme hyperparam tuning
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5,0.999))
    criterion = nn.CrossEntropyLoss()

    model_info = {'best_acc':0, 'best_epoch':-1, 'dargs':dargs}
    loss_info = {'CEloss': [], 'mabloss': []}

    for epoch in range(dargs['n_epoch']):
        shardlist = os.listdir(DIRS['SHARD_FOLDER_DIR'])
        random.shuffle(shardlist)
        net.train()

        nshards = len(shardlist)
        for ns, SHARD_NAME in enumerate(shardlist):
            if (ns+1)==nshards or (nshards+1)%4==0:
                update_text = f'epoch {epoch}: {ns+1}/{nshards}'
                print('%-64s'%(update_text), end='\r')                
            if not SHARD_NAME[-5:] == 'shard': continue

            
            SHARD_DIR = os.path.join(DIRS['SHARD_FOLDER_DIR'], str(SHARD_NAME))
            traindataset = load_dataset_from_a_shard(SHARD_DIR, reshape_size=None)
            if dargs['n_classes'] == 3:
                traindataset = ThreeClassesPyIOwithHeatmap(traindataset.x,traindataset.y, traindataset.h)
            elif dargs['n_classes'] == 10:
                traindataset = TenClassesPyIOwithHeatmap(traindataset.x,traindataset.y, traindataset.h)
            else:
                raise NotImplementedError()
            trainloader = DataLoader(traindataset,batch_size=dargs['batch_size'],  )            
            
            for i,(x,y0,h0) in enumerate(trainloader):
                net.zero_grad()

                x = x.to(torch.float).to(device=device)
                y0 = y0.to(torch.long).to(device=device)
                h0 = h0.to(torch.long).to(device=device)

                y, h = net(x)

                CEloss = criterion(y,y0)
                loss_info['CEloss'].append(CEloss.item())

                if dargs['loss'] == 'CE':
                    CEloss.backward()
                    optimizer.step()
                elif dargs['loss'] == 'CE+mab':
                    b, h_t, w_t = h0.shape
                    mabloss = criterion(h[:,:,:h_t,:w_t], h0)

                    loss = CEloss + dargs['mab_regularization'] * mabloss
                    loss_info['mabloss'].append(mabloss.item())

                    loss.backward()
                    optimizer.step()

            if DEBUG_TOGGLES['train_iter']: break
        print()
        model_info = validate(epoch, net, DIRS, dargs, model_info)

        if model_info['best_acc'] >= _TARGET_ACC_:
            if epoch > dargs['min_epoch']: 
                print('Early stopping achieved. Best val acc:', model_info['best_acc'])
                break 

    with open(DIRS['LOSS_INFO_DIR'], 'w') as jfile:
        json.dump(loss_info, jfile, indent=4, sort_keys=True)


def validate(epoch, net, DIRS, dargs, model_info):
    shardlist = os.listdir(DIRS['SHARD_VAL_FOLDER_DIR'])
    net.eval()

    nshards = len(shardlist)
    for ns,SHARD_NAME in enumerate(shardlist):
        if (ns+1)==nshards or (ns+1)%4==0:
            update_text = f'epoch {epoch}: {ns+1}/{nshards}'
            print('%-64s'%(update_text), end='\r')
        if not SHARD_NAME[-5:] == 'shard': continue

        SHARD_DIR = os.path.join(DIRS['SHARD_VAL_FOLDER_DIR'], str(SHARD_NAME), )
        valdataset = load_dataset_from_a_shard(SHARD_DIR, reshape_size=None)
        valloader = DataLoader(valdataset,batch_size=dargs['batch_size'],  ) 

        total_n_correct, total_n = 0, 0
        for i,(x,y0) in enumerate(valloader):
            with torch.no_grad():
                x = x.to(torch.float).to(device=device)
                y0 = y0.to(torch.long).to(device=device)

                y, h = net(x)
                pred = torch.argmax(y,dim=1)
                n_correct = torch.sum(y0==pred).item()

                total_n_correct += n_correct
                total_n += len(y)

        if DEBUG_TOGGLES['val_iter']: break

    acc = total_n_correct/total_n
    if acc > model_info['best_acc']:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
        }, DIRS['MODEL_DIR'])

        model_info.update({'best_acc': acc, 'best_epoch': epoch})
        with open(DIRS['MODEL_INFO_DIR'], 'w') as json_file:
                json.dump(model_info, json_file, indent=4, sort_keys=True)
        print('%-64s'%(str(f'acc:{acc} (saved)')) )  
    else:
        print('%-64s'%(str(f'acc:{acc}')) )  

    return model_info  


def sample_heatmaps(dargs):
    DIRS = manage_dir(dargs)
    if dargs['n_classes'] == 3:
        net = mabSPA().to(device=device)
    elif dargs['n_classes'] == 10:
        net = mabSPA(fc_output_c=10).to(device=device)
    else:
        raise NotImplementedError()    
    checkpoint = torch.load(DIRS['MODEL_DIR'])

    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()    

    bsize = 8

    shardlist = os.listdir(DIRS['SHARD_VAL_FOLDER_DIR'])
    for ns,SHARD_NAME in enumerate(shardlist):
        SHARD_DIR = os.path.join(DIRS['SHARD_VAL_FOLDER_DIR'], str(SHARD_NAME), )
        valdataset = load_dataset_from_a_shard(SHARD_DIR, reshape_size=None)
        valdataset = ThreeClassesPyIOwithHeatmap(valdataset.x,
            valdataset.y, valdataset.h)
        valloader = DataLoader(valdataset,batch_size=bsize,  ) 

        for i,(x,y0, h0) in enumerate(valloader):
            with torch.no_grad():
                x = x.to(torch.float).to(device=device)
                y0 = y0.to(torch.long).to(device=device)
                y, h = net(x)

                b,c,h_,w_ = x.shape
                imgs = x.clone().detach().cpu().numpy()

                h1 = torch.argmax(h[:,:,:h_,:w_], dim=1)
                h1 = h1.clone().detach().cpu().numpy()
                hpred = (h1==1)*0.4 + (h1==2)*0.9
                h0 = (h0==1)*0.4 + (h0==2)*0.9

                plt.figure(figsize=(12,5))
                for i in range(bsize):
                    plt.gcf().add_subplot(3,8,i+1)
                    plt.gca().imshow(imgs[i].transpose(1,2,0))
                    if i==0: plt.gca().set_ylabel('img')
                    plt.gcf().add_subplot(3,8,i+1+8)
                    plt.gca().imshow(h0[i], cmap='bwr', vmin=-1.,vmax=1.)
                    if i==0: plt.gca().set_ylabel('gt')
                    plt.gcf().add_subplot(3,8,i+1+8*2)
                    plt.gca().imshow(hpred[i], cmap='bwr', vmin=-1.,vmax=1.)
                    if i==0: plt.gca().set_ylabel('heatmaps pred')
                plt.tight_layout()
                plt.savefig(DIRS['HEATMAP_SAMPLE_DIR'])

            break
        break