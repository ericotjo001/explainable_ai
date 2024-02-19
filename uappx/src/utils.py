import numpy as np

def parse_bool_from_string(bool_string):
    # assume bool_string is either 0 or 1 (str)
    if str(bool_string)=='1': return True
    elif str(bool_string)=='0': return False
    else: raise RuntimeError('parse_bool_from_string only accepts 0 or 1.')
strbool_description = 'bool by string 1 or 0 (avoid store_true problem)'

def readjust_bools(args, dargs, BOOLS):
    for bkey in BOOLS:
        adjusted_bool = parse_bool_from_string(dargs[bkey])
        setattr(args, bkey, adjusted_bool)
        dargs[bkey] = adjusted_bool
        # print(bkey, 'true') if dargs[bkey] else print(bkey, 'false')
        # print(bkey, 'true') if getattr(args,bkey) else print(bkey, 'false')
    return args, dargs


def binarize(y, n_class): # for discrete/categorical output
    # n_class is an int
    # y is an int. 0,1,2...,n_class-1

    y_binary = np.zeros(shape=(n_class,))
    y_binary[y]=1.
    return y_binary

#################
# layer activation utils
#################


def supergauss(x,a,n=4):
    x = np.clip(x,-100.,100.)
    return np.exp(-(x/a)**(2*n))

def selective_activation(x,a=0.001):
    return a/(a+x**2)

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


    x = np.linspace(-1,1,240)
    a1, a2, r = 0.1, 0.5, 0.2
    dsa = double_selective_activation(x,a1,a2,r=r)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x,dsa)
    plt.show()