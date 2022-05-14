import numpy as np
from dbscan import DBSCAN
from utils import print_me, myplot
import pandas as pd
import sklearn.datasets as datasets
import scipy.io as scio
from sklearn import preprocessing
from matplotlib.collections import LineCollection
from rho import density_akd, density_fkd, density_naive, density_lc, density_rnn
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.linalg
import scipy.io
from utils import pdist2
import scipy.cluster.vq
from sklearn.neighbors import NearestNeighbors

filename_list = ['gen2', 'gen3']
filename = filename_list[0]
anchor_dict = {
    'type': 'kmeans',  # random, kmeans, kmeanspp, mbkmeans, uniform, leader
    'm_ratio': 0.2,
    's': 2,
    'max_iter': 10,
    'mini_batch': 20,
    'delta': 0.001
}

data = np.loadtxt('data/'+filename+'_data.txt')
label_true = list(np.loadtxt('data/'+filename+'_label.txt'))
data = np.array(data, dtype='float32')
print('* * * * * * * * * *')
print('Data have been loaded.')
print('* * * * * * * * * *')

n = data.shape[0]
m = int(n * anchor_dict['m_ratio'])
s = anchor_dict['s']
anchors = scipy.cluster.vq.kmeans2(data=data, k=m, iter=anchor_dict['max_iter'], minit='points')[0]
X = data
anchors = anchors.astype('float32')
n = X.shape[0]
m = anchors.shape[0]
sqdist = pdist2(X, anchors, 'sqeuclidean')
val = np.zeros((n, s), dtype=np.float32)
pos = np.zeros((n, s), dtype=np.int)
for i in range(s):
    pos[:, i] = np.argmin(sqdist, 1)
    val[:, i] = sqdist[np.arange(len(sqdist)), pos[:, i]]
    sqdist[np.arange(n), pos[:, i]] = float('inf')
    dist = np.sqrt(val[:, s - 1])
    sigma = np.mean(dist) / np.sqrt(2)
c = 2 * np.power(sigma, 2)  # bandwidth parameter
exponent = -val / c  # exponent of RBF kernel
val = np.exp(exponent)
z = scipy.sparse.lil_matrix((n, m), dtype='float32')
for i in range(s):
    z[np.arange(n), pos[:, i]] = val[:, i]
z = z.tocsr()
w = z.dot(z.T)
w = np.asarray(w.todense(), dtype=np.float32)
z = np.asarray(z.todense(), dtype=np.float32)

k = 10
kgraph = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(data).kneighbors_graph(data).astype('int')
kgraph = np.asarray(kgraph.todense())

# w_ = w.copy()
# for i in range(n):
#     for j in range(n):
#         if w[i, j] > 0:
#             w_[i, j] = 1


pltw = kgraph
cnt = np.count_nonzero(pltw)
seg = np.zeros((cnt, 2, 2))
w_seg = np.zeros(cnt)
cnt = 0
for i in range(n):
    for j in range(i + 1, n):
        if pltw[i, j] > 0:
            seg[cnt, :, 0] = np.array([data[i, :][0], data[j, :][0]])
            seg[cnt, :, 1] = np.array([data[i, :][1], data[j, :][1]])
            w_seg[cnt] = pltw[i, j]
            cnt = cnt + 1
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], marker='.')
line = LineCollection(seg)
line.set_array(w_seg)
ax.add_collection(line)
line.set_color('black')
plt.axis = 'off'
plt.xticks([])
plt.yticks([])
plt.show()

# data0 = np.concatenate([data, anchors], axis=0)
# w1 = np.concatenate([np.zeros([m, m]), z], axis=0)
# w2 = np.concatenate([z.T, np.zeros([n, n])], axis=0)
# w_ = np.concatenate([w1, w2], axis=1)
# pltw = w_
# cnt = np.count_nonzero(pltw)
# seg = np.zeros((cnt, 2, 2))
# w_seg = np.zeros(cnt)
# cnt = 0
# for i in range(n+m):
#     for j in range(i + 1, n+m):
#         if pltw[i, j] > 0:
#             seg[cnt, :, 0] = np.array([data0[i, :][0], data0[j, :][0]])
#             seg[cnt, :, 1] = np.array([data0[i, :][1], data0[j, :][1]])
#             w_seg[cnt] = pltw[i, j]
#             cnt = cnt + 1
# fig, ax = plt.subplots()
# ax.scatter(data0[:, 0], data0[:, 1])
# line = LineCollection(seg, colors='gray')
# line.set_array(w_seg)
# ax.add_collection(line)
# plt.axis = 'off'
# plt.xticks([])
# plt.yticks([])
# plt.show()

