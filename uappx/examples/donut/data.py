import numpy as np
import matplotlib.pyplot as plt
import os

# Simple data with 2D X and arbitary label Y
class Data2DInput(object):
    def __init__(self, ):
        super(Data2DInput, self).__init__()

    def show_fig_and_exit(self):
        # N, Dy = 
        ys = self.Y.shape

        lys = len(ys)
        if lys==1:
            N = ys
            self.show()
        elif lys==2:
            N, Dy = ys  
            self.show_at_most_three()
        else:
            raise NotImplementedError()    
        exit()

    def show(self):
        plt.figure()

        y = self.Y
        im = plt.gca().scatter(self.X[:,0],self.X[:,1], s=5, c=y, cmap='jet', )
        if not self.X_test is None:
            y_test = self.Y_test
            im = plt.gca().scatter(self.X_test[:,0],self.X_test[:,1], s=24, c=y_test , cmap='jet', 
                marker='x',alpha=0.4)

        plt.title('only showing the first 2 dims')
        plt.colorbar(im)
        plt.show()



class DonutDataX(Data2DInput):
    def __init__(self, N, test_sd=0.01,
        show_fig_and_exit=True, label_mode='scalar'):
        super(DonutDataX, self).__init__()
        self.N = N

        self.test_sd = test_sd

        T = np.random.uniform(0,2*np.pi*(1-1/N),size=(N))
        R = np.random.uniform(0.8,1.2,size=(N,))
        self.get_data(T,R)
        self.get_labels(T, label_mode=label_mode)

        if label_mode=='discrete':
            self.n_class = 3 # just in case

        if show_fig_and_exit:
            self.show_fig_and_exit()

    def get_data(self, T, R):
        test_sd = self.test_sd

        X = np.array([np.cos(T), np.sin(T)]).T 
        X[:,0] *= R
        X[:,1] *= R
        # print(X.shape) # (N,D=2)
        self.X = X
        self.X_test = self.X + np.random.uniform(-test_sd,test_sd,size=self.X.shape) 
        # print(self.X_test)
        # raise Exception('gg')

    def get_labels(self, T, label_mode='scalar'):
        if label_mode=='scalar':
            self.Y = np.cos(T)
            self.Y_test = np.cos(T)
        elif label_mode in ['multidim','discrete']:
            self.Y = np.stack((np.cos(T),np.cos(T*2),np.cos(T*3))).T # shape (N, 3)
            self.Y_test = np.stack((np.cos(T),np.cos(T*2),np.cos(T*3))).T

        if label_mode == 'discrete':
            self.Y = np.argmax(self.Y,axis=1)
            self.Y_test = np.argmax(self.Y_test,axis=1)

    def save_data_by_class(self,SAVE_DIR):
        # this is to mimic folders like ImageNet
        for i, (x,y) in enumerate(zip(self.X, self.Y)):
            class_folder = os.path.join( SAVE_DIR ,f'class{y}')
            os.makedirs( class_folder, exist_ok=True)
            np.save(os.path.join(class_folder, f'{i}.npy'), x)

    def save_testdata_by_class(self,SAVE_DIR):
        # this is to mimic folders like ImageNet
        for i, (x,y) in enumerate(zip(self.X_test, self.Y_test)):
            class_folder = os.path.join( SAVE_DIR ,f'class{y}')
            os.makedirs( class_folder, exist_ok=True)
            np.save(os.path.join(class_folder, f'{i}.npy'), x)


class DonutDataY(DonutDataX):
    def __init__(self,  *args, **kwargs):
        super(DonutDataY, self).__init__(*args, **kwargs)

    def get_data(self, T, R):
        test_sd = self.test_sd

        X = np.array([np.cos(T), np.sin(T), np.sin(2*T)]).T 
        X[:,0] *= R
        X[:,1] *= R
        X[:,2] *= R
        # print(X.shape) # (N,D=2)
        self.X = X
        self.X_test = self.X + np.random.uniform(-test_sd,test_sd,size=self.X.shape) 

    def get_labels(self, T, label_mode='scalar'):
        if label_mode=='scalar':
            self.Y = np.cos(T)
            self.Y_test = np.cos(T)
        elif label_mode in ['multidim','discrete']:
            self.Y = np.stack((np.cos(T),np.cos(T*2),np.cos(T*3),np.cos(T*4),np.cos(T*5))).T # shape (N, 3)
            self.Y_test = np.stack((np.cos(T),np.cos(T*2),np.cos(T*3),np.cos(T*4),np.cos(T*5))).T

        if label_mode == 'discrete':
            self.Y = np.argmax(self.Y,axis=1)
            self.Y_test = np.argmax(self.Y_test,axis=1)