from multiprocessing import Pool
import time
import numpy as np


def f(x,y,z,c=0,d=0):
    out = x*x+y+z**3+c+d
    timelag = np.random.randint(4)
    time.sleep(timelag)
    print('f():%s [time lag:%s]'%(str(out),str(timelag)))
    return out

def fwrap(args):
    return f(args[0],args[1],args[2],c=args[3],d=args[4])

if __name__ == '__main__':
    items = [[1,2,3,100,10000],
    [3,4,5,200,20000]]

    for i in range(3):
        t1 = time.time()
        p = Pool(4)
        all_out = p.map(fwrap, items)
        t2 = time.time()
        p.close()
        p.join()
        # this is to make sure that we wait till 
        # the process in this iteration ends first
        print("time taken = " + str(t2 - t1))
        print("all_out:\n",all_out)
        print()
