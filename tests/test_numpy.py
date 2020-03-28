import numpy as np

print(np.zeros(shape=(2,2)))

print(np.max([1.0,2.0]))

print(np.mean([]))

x = []
for i in range(3):
	x.append(np.random.normal(0,1,size=(4,)))

print(np.concatenate(x).shape)