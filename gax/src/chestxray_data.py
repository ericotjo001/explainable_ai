from .utils import *
from skimage.transform import resize
from torch.utils.data import Dataset

def resize_numpy_img_array(x, target_size, dims='HWC'):
    if dims=='HWC':
        """
        target_size: (H,W,C)
        """
        # HWC is height,weight, channel=3
        return resize(x, target_size)
    elif dims=='CHW':
        """
        target_size: (C,H,W)
        """
        x = x.transpose(1,2,0) # becomes HWC
        x = resize(x, (target_size[1],target_size[2],target_size[0]))
        x = x.transpose(2,0,1) # back to CWH
        return x
    else:
        raise RuntimeError('Invalid dims.')

class chestXRayPneumonia(Dataset):
    # for https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    def __init__(self, DATA_DIR, split='train', img_size=(256,256)):
        super(chestXRayPneumonia, self).__init__()
        self.DATA_DIR = DATA_DIR
        self.split = split
        self.img_size = img_size

        self.n = 0
        self.data_name_and_label = []
        self.LABELS = ['NORMAL', 'PNEUMONIA']
        for y0, label in enumerate(self.LABELS):
            LIST_OF_DATA = os.path.join(DATA_DIR, split, label)
            for img_file_name in os.listdir(LIST_OF_DATA):
                self.data_name_and_label.append((img_file_name, y0))
                self.n += 1

    def simple_img_preprocessing(self, pil_img):
        #
        # THIS WHOLE THING SEEMS INEFFICIENT. Is tehre any faster way?
        # 

        img = np.asarray(pil_img) 
        # self.show_img(img)
        if len(img.shape)==2: 
            # print(img.shape) # only H, W
            img = np.array([img.T,img.T,img.T]) # C,H,W
        else: 
            # print(img.shape) # it is loaded in H,W,C
            img = img.transpose(2,1,0)

        img = resize_numpy_img_array(img, target_size= (3,) + self.img_size, dims='CHW')
        # print(img.shape) # (1,960,960)
        # self.show_img(img[0].T)
        return img  

    def __getitem__(self, i):
        img_file_name, y0 = self.data_name_and_label[i]
        img_dir = os.path.join(self.DATA_DIR, self.split, self.LABELS[y0], img_file_name)        

        pil_img = Image.open(img_dir)

        # x = resize(np.array(pil_img), self.img_size)
        # # print(np.array(pil_img).shape, x.shape) # (1317, 1857) (256, 256)
        # # print(np.max(x), np.min(x)) # 0.9529411764705882 0.0424188202502666

        # x = np.stack((x,x,x))
        # print(x.shape) # (3,256,256)
        x = self.simple_img_preprocessing(pil_img)

        return x, y0

    def __len__(self,):
        return self.n



class chestXRayCovid(Dataset):
    # https://www.kaggle.com/code/sana306/detection-of-covid-positive-cases-using-dl
    def __init__(self, DATA_DIR, 
                DATA_SPLIT_CACHE, 
                split='train',
                img_size=(256,256)):                
        super(chestXRayCovid, self).__init__()
        self.DATA_DIR = DATA_DIR
        self.DATA_SPLIT_CACHE = DATA_SPLIT_CACHE
        self.split = split
        self.img_size = img_size

        self.LABELS = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']

        self.populate_data_list_by_split()
        self.n = len(self.data_list[self.split])

    def populate_data_list_by_split(self):
        if not os.path.exists(self.DATA_SPLIT_CACHE):
            self.data_list = {
                'train': [],
                'val': [],
                'test': []
            }

            for y0, LABEL in enumerate(self.LABELS):
                img_folder_dir = os.path.join(self.DATA_DIR, LABEL, 'images' )
                for j,imgname in enumerate(os.listdir(img_folder_dir)):
                    if j%3==0:
                        this_split = 'train'
                    elif j%3==1:
                        this_split = 'val'
                    else:
                        this_split = 'test'
                    self.data_list[this_split].append((imgname, y0))

            joblib.dump(self.data_list, self.DATA_SPLIT_CACHE)
            print(f'saving cache to {self.DATA_SPLIT_CACHE}')
        else:
            print(f'loading cache from {self.DATA_SPLIT_CACHE}')
            self.data_list = joblib.load(self.DATA_SPLIT_CACHE)
        
    def __len__(self):
        return self.n

    def __getitem__(self,i):
        imgname, y0 = self.data_list[self.split][i]

        img_dir = os.path.join(self.DATA_DIR,self.LABELS[y0], 'images', imgname)

        x = np.array(Image.open(img_dir))
        if len(x.shape)>2:
            x = np.mean(x, axis=2)
        x = np.expand_dims(x, axis=0)/255.
        return x, y0