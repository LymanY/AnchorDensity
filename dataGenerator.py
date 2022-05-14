import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.datasets import make_circles


# n1 = int(2000/2)
# n2 = int(500/2)
# n3 = int(1500/2)
# n4 = int(1500/2)
# # c1 = np.concatenate((np.random.rand(n1, 1)*50+5, np.random.rand(n1, 1)*5+45), axis=1)
# c1 = np.concatenate((np.random.randn(n1, 1)*10+25, np.random.randn(n1, 1)*1+50), axis=1)
# c2 = np.concatenate((np.random.randn(n2, 1)*8+25, np.random.randn(n2, 1)*3+35), axis=1)
# c3 = np.concatenate((np.random.randn(n3, 1)*4+12, np.random.randn(n3, 1)*2+18), axis=1)
# c4 = np.concatenate((np.random.randn(n4, 1)*4+34, np.random.randn(n4, 1)*2+18), axis=1)
#
# data = np.concatenate((c1, c2, c3, c4), axis=0)
# label = np.concatenate((np.zeros(n1)+0, np.zeros(n2)+1, np.zeros(n3)+2, np.zeros(n4)+3)).reshape(data.shape[0], 1).astype(int)
#
# filename = 'test'
# np.savetxt('data/'+filename+'_data.txt', data, fmt='%f')
# np.savetxt('data/'+filename+'_label.txt', label, fmt='%d')
#
# plt.scatter(data[:, 0], data[:, 1], c=label)


# n1 = int(500)
# n2 = int(1000)
# n3 = int(1000)
# n4 = int(1000)
#
#
# x1, y1 = make_circles(n_samples=n1*2, factor=0.8, shuffle=False, noise=0.06)
# c1 = np.concatenate((x1[:, 0].reshape(n1*2, 1)*40+30, x1[:, 1].reshape(n1*2, 1)*40+20), axis=1)[0:n1, :]
# c2 = np.concatenate((np.random.randn(n2, 1)*3+25, np.random.randn(n2, 1)*3+35), axis=1)
# c3 = np.concatenate((np.random.randn(n3, 1)*3+15, np.random.randn(n3, 1)*3+18), axis=1)
# c4 = np.concatenate((np.random.randn(n4, 1)*3+36, np.random.randn(n4, 1)*3+18), axis=1)
#
#
# data = np.concatenate((c1, c2, c3, c4), axis=0)
# label = np.concatenate((np.zeros(n1)+0, np.zeros(n2)+1, np.zeros(n3)+2, np.zeros(n4)+3)).reshape(data.shape[0], 1).astype(int)
#
# filename = 'test'
# np.savetxt('data/'+filename+'_data.txt', data, fmt='%f')
# np.savetxt('data/'+filename+'_label.txt', label, fmt='%d')
#
# plt.scatter(data[:, 0], data[:, 1], c=label)



n1 = int(2000*2)
n2 = int(500*2)
n3 = int(1500*2)
n4 = int(1500*2)
# c1 = np.concatenate((np.random.rand(n1, 1)*50+5, np.random.rand(n1, 1)*5+45), axis=1)
c1 = np.concatenate((np.random.randn(n1, 1)*10+25, np.random.randn(n1, 1)*1+50), axis=1)
c2 = np.concatenate((np.random.randn(n2, 1)*8+25, np.random.randn(n2, 1)*3+35), axis=1)
c3 = np.concatenate((np.random.randn(n3, 1)*4+12, np.random.randn(n3, 1)*2+18), axis=1)
c4 = np.concatenate((np.random.randn(n4, 1)*4+34, np.random.randn(n4, 1)*2+18), axis=1)

data = np.concatenate((c1, c2, c3, c4), axis=0)
data = np.concatenate((data, np.zeros((11000, 2498))), axis=1)
label = np.concatenate((np.zeros(n1)+0, np.zeros(n2)+1, np.zeros(n3)+2, np.zeros(n4)+3)).reshape(data.shape[0], 1).astype(int)

filename = 'd2500'
np.savetxt('data/'+filename+'_data.txt', data, fmt='%f')
np.savetxt('data/'+filename+'_label.txt', label, fmt='%d')

# plt.scatter(data[:, 0], data[:, 1], c=label)