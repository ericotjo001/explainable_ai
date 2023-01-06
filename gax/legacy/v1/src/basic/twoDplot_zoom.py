import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

def get_change_of_basis(theta):
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    Rinv = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    return R, Rinv

def get_w(theta, a1, eta):
    a2 = 1- a1
    w = np.array([[1.],[1.]])
    dw = eta* np.array([
        [(np.cos(theta)+np.sin(theta))*(a1*np.cos(theta)-a2*np.sin(theta))],
        [(np.sin(theta)-np.cos(theta))*(a1*np.sin(theta)+a2*np.cos(theta))]]
    )
    return w+ eta* dw

def get_x(a1,theta):
    R11 = np.cos(theta)
    R22 = np.cos(theta)
    R12 = -np.sin(theta)
    R21 = np.sin(theta)
    x1 = a1*R11 + (1-a1)*R12
    x2 = a1*R21 + (1-a1)*R22
    return x1, x2

font = {'size': 16}
plt.rc('font', **font)


eta =1.2
A1 = [0.95,0.7,0.55]
SETTINGS = {
    0.95: {
        'xticks':[0, np.pi/4,np.pi/2],
        'xtickslabels':['0','$\dfrac{\pi}{4}$','$\dfrac{\pi}{2}$'],
        'xlim':[0,np.pi/2],
        }, 
    0.7: {
        'xticks':[-np.pi/8, 0, np.pi/4, 3*np.pi/8],
        'xtickslabels':['-$\dfrac{\pi}{8}$','0', '$\dfrac{\pi}{4}$','$\dfrac{3\pi}{8}$'],
        'xlim':[-np.pi/8, 3*np.pi/8],
        },
    0.55: {
        'xticks':[-np.pi/4,0,np.pi/4,],
        'xtickslabels':['-$\dfrac{\pi}{4}$','0','$\dfrac{\pi}{4}$'],
        'xlim':[-np.pi/4,np.pi/4,],
        },
}
plt.figure(figsize=(12,5))
for i,a1 in enumerate(A1):
    # a1 = 0.9
    X1, X2 = [],[]
    ATTR1, ATTR2 = [], []
    THETA = np.linspace(-np.pi, np.pi, 128)
    for theta in THETA:
        x1,x2 = get_x(a1,theta)
        w1,w2 = get_w(theta,a1, eta)
        attr1, attr2 = x1*w1, x2*w2
        R, Rinv = get_change_of_basis(theta)
        X1.append(x1)
        X2.append(x2)
        ATTR1.append(attr1)
        ATTR2.append(attr2)
    plt.gcf().add_subplot(1,len(A1),i+1)
    plt.gca().plot(THETA, X1, c='r', label='$x_1$', linewidth=0.5)
    plt.gca().scatter(THETA, ATTR1, s=1, c='r', label='$h_1$')
    plt.gca().plot(THETA ,X2, c='b', label='$x_2$', linewidth=0.5)
    plt.gca().scatter(THETA, ATTR2, s=1, c='b', label='$h_2$')
    plt.gca().set_xticks(SETTINGS[a1]['xticks'])
    plt.gca().set_xticklabels(SETTINGS[a1]['xtickslabels'])
    plt.gca().set_xlim(SETTINGS[a1]['xlim'])
    plt.gca().set_title('$a_1=%.3f, a_2=%.3f$'%(a1,1-a1))

    for x in [-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4,np.pi/2, 3*np.pi/4, np.pi]:
        plt.gca().vlines(x,-2.4,2.4, linestyles='dashed',colors='k', linewidth=0.5)
    plt.gca().set_xlabel('$\\theta$')
    if i==0:
        plt.legend()
plt.tight_layout()
plt.show()