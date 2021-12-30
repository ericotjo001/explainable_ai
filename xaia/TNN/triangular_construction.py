import numpy as np

def setup_one_dim_triangular(x, y, a=5.):
    # x is array of shape (N,) input
    # y is array of shape (N,) target variable
    x1 = x*0
    x1[:-1] = x[1:]
    delta = x - x1
    delta[-1] = np.nanmean(delta[:-1])

    W = 2*a/delta[::-1]
    b = a - W * x[::-1]

    if y is not None:
        Ainv = setup_inverse_A_matrix(len(x))
        alpha = np.matmul(Ainv, y.reshape(-1,1))   
    else:
        return W.reshape(-1,1), b.reshape(-1,1)

    return W.reshape(-1,1), b.reshape(-1,1), alpha

def setup_inverse_A_matrix(ndim):
    Ainv = np.zeros(shape=(ndim,ndim))
    for i in range(ndim):
        Ainv[i,ndim-1-i] = 1
    for i in range(1,ndim):
        Ainv[i,ndim-i] = -1
    return Ainv

def approx_function(x, W, b, alpha):
    # x is scalar
    sig = activations(x,W,b)
    return  np.matmul(alpha.T, sig).reshape(-1)[0]

def activations(x, W, b):
    temp = np.clip(W*x+b,-100.,100.)
    sig = 1./(1.+np.exp(-temp))
    return sig