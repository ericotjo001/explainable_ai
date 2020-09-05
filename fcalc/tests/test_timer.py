import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir

from utils.utils import *
# start = time.time()
# start2 = time.time()

# time.sleep(1.) # in seconds

# end = time.time()
# time.sleep(2.0)
# end2 = time.time()

# elapsed = end - start
# print('time taken   [s]:%s'%(str(elapsed))) 
# elapsed2 = end2 - start2
# print('time taken 2 [s]:%s'%(str(elapsed2))) 

ckpt1 = time.time()
time.sleep(1.)
ckpt2 = time.time()
time.sleep(3.)
ckpt3 = time.time()
time.sleep(5.)
ckpt4 = time.time()

print(ckpt2-ckpt1)
print(ckpt3-ckpt1)
print(ckpt4-ckpt2)