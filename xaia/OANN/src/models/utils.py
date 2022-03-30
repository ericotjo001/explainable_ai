import numpy as np


def binarize(y, n_class): # for discrete/categorical output
    # n_class is an int
    # y is an int. 0,1,2...,n_class-1

    y_binary = np.zeros(shape=(n_class,))
    y_binary[y]=1.
    return y_binary

def supergauss(x,a,n=4):
    x = np.clip(x,-100.,100.)
    return np.exp(-(x/a)**(2*n))

def selective_activation(x,a=0.001):
    return a/(a+x**2)

# def double_selective_activation(x,r=0.5, a1=1e-4, a2=1.):
#     """
#     x = np.linspace(-2.,2.,480)
#     y = double_selective_activation(x)
#     plt.figure()
#     plt.plot(x,y)
#     plt.show()
#     """
#     out = (1-r)*selective_activation(x,a1) + r*supergauss(x,a2)
#     return np.clip(out, 0.,1.) 

def double_selective_activation(x, a1, a2, r=0.5):
    """
    import matplotlib.pyplot as plt
    x = np.linspace(-2.,2.,480)
    a1 = 1e-4 + np.zeros(shape=x.shape)
    a2 = 1. + np.zeros(shape=x.shape)
    y = double_selective_activation(x,a1,a2)
    plt.figure()
    plt.plot(x,y)
    plt.show()
    """
    out = (1-r)*selective_activation(x,a1) + r*supergauss(x,a2)
    return np.clip(out, 0.,1.) 

if __name__=='__main__':
    print('testing utils')