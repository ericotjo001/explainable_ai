import os, pickle
from .printing_manager import ShortPrint
sp = ShortPrint() 



def manage_directories(args):
    # assume the root dir is already changed to gax folder
    DIRS = {}
    DIRS['ROOT_DIR'] = args['ROOT_DIR']
    DIRS['CHECKPOINT_DIR'] = 'checkpoint' 
    create_if_not_exists(DIRS['CHECKPOINT_DIR'])
    DIRS['PROJECT_DIR'] = os.path.join(DIRS['CHECKPOINT_DIR'],args['PROJECT_ID'])
    create_if_not_exists(DIRS['PROJECT_DIR'])
    DIRS['MODEL_DIR'] = os.path.join(DIRS['PROJECT_DIR'], 'main.model')
    # DIRS['SAVE_IMG_FOLDER'] = os.path.join(DIRS['PROJECT_DIR'],'existing_methods')
    # create_if_not_exists(DIRS['SAVE_IMG_FOLDER'])

    DIRS['RESULT_DIR'] = os.path.join(DIRS['PROJECT_DIR'],'test.result.json')

    # assume data is located in gax/data 
    DIRS['DATA_DIR'] =  os.path.join('data','chest_xray')
    return DIRS

def create_if_not_exists(this_dir):
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)

def check_default_aux_folders():
    DEFAULT_AUX_FOLDER = ['checkpoint', 'checkpoint/cache']
    for folder_name in DEFAULT_AUX_FOLDER:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        else:
            pass # print(folder_name,'OK')

class FastPickleClient(object):
    def __init__(self):
        super(FastPickleClient, self).__init__()
        self.save_text = 'Saving data via FastPickleClient...'
        self.load_text = 'Loading data via FastPickleClient...'
    
    def pickle_data(self, save_data, save_dir, tv=(0,0,None), text=None):
        if text is not None: 
            self.save_text = text
        output = open(save_dir, 'wb')
        pickle.dump(save_data, output)
        output.close()      
        sp.prints('%s\n  %s'%(str(self.save_text),str(save_dir)), tv=tv)

    def load_pickled_data(self, pickled_dir, tv=(0,0,None), text=None):
        if text is not None:
            self.load_text = text
        pkl_file = open(pickled_dir, 'rb')
        this_data = pickle.load(pkl_file)
        pkl_file.close()        
        sp.prints('%s\n  %s'%(str(self.load_text),str(pickled_dir)), tv=tv)
        return this_data

fastpickle = FastPickleClient()


###################### resize pack ##############################

import numpy as np
from skimage.transform import resize

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

def fit_center_aspect_invariant_resize_HWC(x, target_size):
    """
    target_size: (H,W,C)
    """
    target_aspect = target_size[0]/target_size[1]
    h,w,c = x.shape
    if h>=w:
        w1 = int(h/target_aspect)
        holder = np.zeros(shape=(h, w1,c))
        horizontal_shift = int((w1-w)/2)
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    holder[i][int(j+ horizontal_shift)][k] = x[i][j][k] 
    else:
        h1 = int(w*target_aspect)
        holder = np.zeros(shape=(h1,w,c))
        vertical_shift = int((h1-h)/2)
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    holder[int(i+ vertical_shift)][j][k] = x[i][j][k] 
    y = resize_numpy_img_array(holder,target_size,dims='HWC')
    return y

class GetResizeExamples(object):
    def __init__(self):
        super(GetResizeExamples, self).__init__()

        """
        Example usage:
        gex = GetResizeExamples()
        gex.get_example_resize_numpy_img_array(example=1)
        gex.get_example_fit_center_aspect_invariant_resize_HWC()
        """

    def get_example_resize_numpy_img_array(self, example=0):
        import matplotlib.pyplot as plt
        plt.figure()
        
        if example==0:
            s = (512,512,3)
            plt.gcf().add_subplot(121)
            x = np.random.normal(0,1,size=[int(s[0]/64),int(s[1]/64),3])
            plt.gca().imshow(x)
            plt.gcf().add_subplot(122)
            x = resize_numpy_img_array(x, s, dims='HWC')
            plt.gca().imshow(x)
        elif example==1:
            s = (3,512,512)
            x = np.random.normal(0,1, size=[3,int(s[1]/64),int(s[2]/64)])
            plt.gcf().add_subplot(211)
            plt.gca().imshow(x.transpose(1,2,0))
            plt.gcf().add_subplot(212)
            x = resize_numpy_img_array(x, s, dims='CHW')
            plt.gca().imshow(x.transpose(1,2,0)) # becomes HWC again
        
        plt.show()

    def get_example_fit_center_aspect_invariant_resize_HWC(self):
        import matplotlib.pyplot as plt
        s0 = (400,300,3)
        x = np.zeros(shape=s0)
        c = (int(s0[0]/2), int(s0[1]/2))
        for i in range(s0[0]):
            for j in range(s0[1]):
                x[i][j][0] = 0.01* ((i-c[0])**2+(j-c[1])**2)**0.5
                x[i][j][1] = 0.04* ((i-c[0])**2+(j-c[1])**2)**0.5
                x[i][j][2] = 0.005* ((i-c[0])**2+(j-c[1])**2)**0.5

        plt.figure()
        plt.gcf().add_subplot(141)
        plt.gca().imshow(x)
        plt.gcf().add_subplot(142)
        s = (400,400,3)
        y = fit_center_aspect_invariant_resize_HWC(x, target_size=s)
        plt.gca().imshow(y)
        plt.gcf().add_subplot(143)
        s = (400,600,3)
        y = fit_center_aspect_invariant_resize_HWC(x, target_size=s)
        plt.gca().imshow(y)
        plt.gcf().add_subplot(144)
        s = (100,80,3)
        y = fit_center_aspect_invariant_resize_HWC(x, target_size=s)
        plt.gca().imshow(y)
        plt.tight_layout()
        plt.show()


