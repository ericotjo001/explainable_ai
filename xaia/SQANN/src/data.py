import numpy as np
import matplotlib.pyplot as plt


def distance_function(X):
    # X shape (N,D)
    return (X[:,0]**2+X[:,1]**2)**0.5


# Simple data with 2D X and arbitary label Y
class Data2DInput(object):
    def __init__(self, ):
        super(Data2DInput, self).__init__()

    def show_fig_and_exit(self):
        plt.figure()
        im = plt.gca().scatter(self.X[:,0],self.X[:,1], s=5, c=self.Y, cmap='jet', )

        if not self.X_test is None:
            im = plt.gca().scatter(self.X_test[:,0],self.X_test[:,1], s=24, c=self.Y_test , cmap='jet', 
                marker='x',alpha=0.4)

        plt.colorbar(im)
        plt.show()
        exit()

class N2LDData(Data2DInput):
    # Noisy 2D Linear Domain data
    def __init__(self, N, vmin=-1., vmax=1.,sd=0.1, test_sd=0.01,
        func='distance',
        show_fig_and_exit=True):
        super(N2LDData, self).__init__()
        self.N = N

        T = np.random.uniform(vmin, vmax, size=(N,))
        temp = np.stack((T,T))
        X = temp + np.random.normal(0,sd,size=temp.shape)
        self.X = X.T
        self.X_test = self.X + np.random.uniform(-test_sd,test_sd,size=self.X.shape) 

        # let's setup the ground-truth values of the function f(X)
        if func=='distance':
            self.Y = distance_function(self.X) 
            self.Y_test = distance_function(self.X_test)
        
        if show_fig_and_exit:
            self.show_fig_and_exit()

class DonutData(Data2DInput):
    def __init__(self, N, test_sd=0.01,
        show_fig_and_exit=True):
        super(DonutData, self).__init__()
        self.N = N

        self.test_sd = test_sd

        T = np.random.uniform(0,2*np.pi*(1-1/N),size=(N))
        R = np.random.uniform(0.8,1.2,size=(N,))
        self.get_data(T,R)
        self.get_labels(T)
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

    def get_labels(self, T):
        self.Y = np.cos(T)
        self.Y_test = np.cos(T)
                 

# if __name__=='__main__':
#     print('show data')
#     N=256
#     ld = N2LDData(N=N, show_fig_and_exit=False)
#     dd = DonutData(N=N, show_fig_and_exit=False)

#     plt.figure(figsize=(8,5))
#     plt.gcf().add_subplot(121)
#     plt.gca().scatter(ld.X[:,0],ld.X[:,1], s=3,c=ld.Y, cmap='jet')
#     plt.gcf().add_subplot(122)
#     plt.gca().scatter(dd.X[:,0],dd.X[:,1], s=3,c=dd.Y, cmap='jet')
#     plt.show()