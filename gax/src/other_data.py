from .utils import *
from torch.utils.data import Dataset

class CreditcardFraudData(Dataset):
    def __init__(self, DATA_DIR):
        super(CreditcardFraudData, self).__init__()
        
        self.DATA_DIR = DATA_DIR
        self.df = pd.read_csv(self.DATA_DIR, float_precision='high', index_col=False)

        self.n = len(self.df)

    def __getitem__(self, i):
        this_row = self.df.loc[i]
        x = np.array(this_row[:-1])
        x = np.clip(x, a_min=-3., a_max=3.)/3. + 0.5
        y0 = int(this_row[-1])
        return x,y0

    def __len__(self,):
        return self.n

class DryBeanDataset(Dataset):
    def __init__(self, DATA_DIR, NORMALIZATION_DIR):
        super(DryBeanDataset, self).__init__()

        self.DATA_DIR = DATA_DIR
        self.df = pd.read_csv(self.DATA_DIR, float_precision='high', index_col=False)

        self.n = len(self.df)        

        with open(NORMALIZATION_DIR) as f:
            norm = json.load(f)
        self.mean = np.array(norm['mean']).astype(np.float64)
        self.std = np.array(norm['std']).astype(np.float64)

        # the following is the class labels in the order found in https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset
        # the order isn't very important, so long as it's consider throughout
        self.CLASSES = [ 'SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON',  'HOROZ', 'SIRA']
        self.CLASS_DICT = { c : y0 for y0,c in enumerate(self.CLASSES)}
        # print(self.CLASS_DICT)
        # {'SEKER': 0, 'BARBUNYA': 1, 'BOMBAY': 2, 'CALI': 3, 'DERMASON': 4, 'HOROZ': 5, 'SIRA': 6}

    def __getitem__(self, i):
        this_row = self.df.loc[i]
        x = np.array(this_row[:-1]).astype(np.float64)
        x = (x-self.mean)/self.std + 0.5
        y0 = self.CLASS_DICT[this_row[-1]]
        return x, int(y0)

    def __len__(self,):
        return self.n