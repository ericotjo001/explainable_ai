import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MnistDataset(object):
    def __init__(self, DIR, train=True):
        super(MnistDataset, self).__init__()
        
        self.dataset = torchvision.datasets.MNIST(root=DIR, train=train, download=False)
        self.n = len(self.dataset)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self,index, ):
        img,y0 = self.dataset.__getitem__(index)
        x = np.array(img)/255
        x = x.reshape((1,)+x.shape)
        return x,y0

def prepare_dataloader(DIR, train=True, batch_size=4, shuffle=True):
    ds = MnistDataset (DIR, train=train,)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader
