from pipeline.training.shared_dependencies import *
from model.adjusted_resnet34 import AdjResnet34

DEBUG_MODE = 0

if DEBUG_MODE:
    DEBUG_VIEW_BATCH_OF_NINE = 0 
    DEBUG_ONE_ITER_ONE_EPOCH = 0 
    DEBUG_N_ITER_TWO_EPOCHS = 8 
else:
    # DO NOT EDIT
    DEBUG_VIEW_BATCH_OF_NINE = 0 # bool
    DEBUG_ONE_ITER_ONE_EPOCH = 0 # bool
    DEBUG_N_ITER_TWO_EPOCHS = 0 # (int) allow N iters per epoch

from pipeline.workflow_config import DEFAULT_CONFIG_DATA
DEFAULT_CONFIG_DATA['model_name'] = 'resnet34adj_0001'
from pipeline.data.prepare_10classes_data import DEFAULT_DATA_CONFIG_DATA
for xkey, x in DEFAULT_DATA_CONFIG_DATA.items():
    DEFAULT_CONFIG_DATA[xkey] = x

if DEBUG_N_ITER_TWO_EPOCHS>0: DEFAULT_CONFIG_DATA['n_epoch'] = 2
if DEBUG_VIEW_BATCH_OF_NINE: DEFAULT_CONFIG_DATA['batch_size'] = 9


def do_DEBUG_ONE_ITER_ONE_EPOCH(DEBUG_ONE_ITER_ONE_EPOCH, x, y0, y, net):
    # print('do_DEBUG_ONE_ITER_ONE_EPOCH')
    DEBUG_SIGNAL = False
    if DEBUG_ONE_ITER_ONE_EPOCH:
        DEBUG_SIGNAL = True
        print('net device: ',next(net.parameters()).device)
        print('x.shape:%s, y.shape:%s , y0.shape:%s'%(str(x.shape),str(y.shape),str(y0.shape)))
    return DEBUG_SIGNAL

def do_DEBUG_VIEW_BATCH_OF_NINE(DEBUG_VIEW_BATCH_OF_NINE, x, y0):
    DEBUG_SIGNAL = False
    if DEBUG_VIEW_BATCH_OF_NINE:
        fig = plt.figure(figsize=(8,8))      
        plt.axis('off')     
        x1 = x.clone().detach().numpy().transpose(0,2,3,1)

        print('  x.transpose(0,2,3,1).shape:',x1.shape, 'y0', y0.clone().detach().numpy())
        for j in range(9):
            if j==0: plt.title(y0)
            fig.add_subplot(331+j)
            plt.gca().imshow(x1[j])

        plt.show()
        DEBUG_SIGNAL = True
    return DEBUG_SIGNAL

