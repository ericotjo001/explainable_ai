import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# put the data in folder data/cifar-10-python.tar.gz if you already downloaded it
class CIFAR10Dataset(Dataset):
    def __init__(self, train=True, root_dir='data'):
        super(CIFAR10Dataset, self).__init__()
        self.dataset =  torchvision.datasets.CIFAR10(root=root_dir, train=train , download=True)
        self.label_to_name = ['airplane','automobile', 'bird','cat','deer','dog','frog','horse', 'ship','truck',]

    def __getitem__(self,index):
        img,y0 = self.dataset.__getitem__(index)
        img = np.array(img).transpose(2,0,1)/255.
        # print(img.shape, y0)
        return img, y0

    def __len__(self):
        return self.dataset.__len__()

    def demo_some_images(self):
        plt.figure(figsize=(8,6))
        for i in range(20):
            plt.gcf().add_subplot(4,5,i+1)
            img, y0 = self.__getitem__(i)
            title = '%s:%s'%(str(y0),str(self.label_to_name[y0]))
            plt.gca().imshow(img.transpose(1,2,0))
            plt.gca().set_title(title)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
        plt.tight_layout()
        plt.show()

def prepare_cifarloader(train=True, root_dir='data', batch_size=4, shuffle=True, demo=False, download=False):
    cif = CIFAR10Dataset(train=train, root_dir=root_dir)
    if demo:  cif.demo_some_images()
    loader = DataLoader(cif, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader

# trainloader = prepare_cifarloader(train=True, root_dir='data', batch_size=4, shuffle=True, demo=1, download=0)
# print('len(trainloader):',len(trainloader))
# for i, data in enumerate(trainloader):
#     x,y = data
#     print('x.shape:',x.shape)
#     print('y:',y)
#     print('x min max: [%s,%s]'%(str(torch.min(x).item()),str(torch.max(x).item())) )
#     break
