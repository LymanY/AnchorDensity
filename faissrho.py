from anchorgraph import AnchorG
import math
import numpy as np
import time
import scipy.cluster.vq
import sklearn.cluster
from sklearn.neighbors import NearestNeighbors
import random
from math import exp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.distance import sqeuclidean
from sklearn.cluster import MiniBatchKMeans
import faiss


def faiss_fkd(data, nn_dict=None):
    n = data.shape[0]
    d = data.shape[1]
    k = nn_dict['k']
    nn_type = nn_dict['type']
    delta = nn_dict['delta']
    st0 = time.time()
    if nn_type in ['kd_tree', 'ball_tree']:
        # neighbor = NearestNeighbors(n_neighbors=k+1, algorithm=nn_type).fit(data).kneighbors_graph(data).toarray()
        index = faiss.IndexFlatL2(d)
        index.add(data)
        _, I = index.search(data, k)

        wk = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(k):
                wk[i, I[i, j]] = exp(sqeuclidean(data[i, :], data[I[i, j], :]) / -0.5)

        # neighbor = np.zeros((n, n))
        # for i in range(n):
        #     neighbor[i][I[i]] = 1
        # wk = np.zeros((n, n), dtype=np.float32)
        # ss = np.where(neighbor != 0)
        # lens = ss[0].size
        # for i in range(lens):
        #     wk[ss[0][i], ss[1][i]] = exp(sqeuclidean(data[ss[0][i], :], data[ss[1][i], :]) / -0.5)

    et0 = time.time()
    print("* * * * * time * * * * *")
    print("Time for computing RHO: %f s" % (et0 - st0))
    tt = et0 - st0
    wk[np.eye(n, dtype=np.bool)] = 0
    wk = wk + delta
    wk = wk / wk.sum(axis=1, keepdims=1)
    rho = np.sum(wk, axis=0)
    return np.array(rho, np.float32), tt

def faiss_naive(data, nn_dict=None):
    n = data.shape[0]
    d = data.shape[1]
    k = nn_dict['k']
    eps = nn_dict['eps']
    nn_type = nn_dict['type']
    st0 = time.time()
    # graph = NearestNeighbors(radius=eps, algorithm=nn_type).fit(data).radius_neighbors_graph(data)
    # rho = np.sum(graph, axis=0)
    # rho = np.array(rho).flatten()
    # rho = np.array(n)
    index = faiss.IndexFlatL2(d)
    index.add(data)
    D, I = index.search(data, k)
    rho = D[:, k-1]
    rho = 1 / rho
    et0 = time.time()
    print("* * * * * time * * * * *")
    print("Time for computing RHO: %f s" % (et0 - st0))
    tt = et0 - st0
    return np.array(rho, np.float32), tt

def faiss_lc(data, nn_dict=None):
    n = data.shape[0]
    d = data.shape[1]
    eps = nn_dict['eps']
    k = nn_dict['k']
    nn_type = nn_dict['type']
    st0 = time.time()
    rho_naive, _ = faiss_naive(data, nn_dict)
    rho = np.zeros(n)
    index = faiss.IndexFlatL2(d)
    index.add(data)
    _, I = index.search(data, k)
    for i in range(n):
        rho[i] = np.count_nonzero(rho_naive[I[i, :].tolist()] < rho_naive[i])
    et0 = time.time()
    print("* * * * * time * * * * *")
    print("Time for computing RHO: %f s" % (et0 - st0))
    tt = et0 - st0
    return np.array(rho, np.float32), tt

def faiss_rnn(data, nn_dict=None):
    n = data.shape[0]
    d = data.shape[1]
    k = nn_dict['k']
    nn_type = nn_dict['type']
    st0 = time.time()
    index = faiss.IndexFlatL2(d)
    index.add(data)
    _, I = index.search(data, k)
    kgraph = np.zeros((n, n))
    for i in range(n):
        kgraph[i][I[i]] = 1
    rho = np.sum(kgraph, axis=0)
    rho = np.array(rho).flatten()
    et0 = time.time()
    print("* * * * * time * * * * *")
    print("Time for computing RHO: %f s" % (et0 - st0))
    tt = et0 - st0
    return np.array(rho, np.float32), tt

def density_akd(data, anchor_dict=None, antype='kmeans'):
    n = data.shape[0]
    d = data.shape[1]
    m = int(n * anchor_dict['m_ratio'])
    s = anchor_dict['s']
    anchor_type = antype
    delta = anchor_dict['delta']
    st0 = time.time()
    if anchor_type == 'random':
        anchor = data[random.sample(range(n), m)]
    elif anchor_type == 'kmeanspp':
        anchor = sklearn.cluster.kmeans_plusplus(data, n_clusters=m, random_state=None)[0]
    elif anchor_type == 'uniform':
        anchor = data[np.linspace(0, n - 1, m, dtype=int)]
    elif anchor_type == 'mbkmeans':
        anchor = MiniBatchKMeans(n_clusters=m, random_state=0,
                                    batch_size=anchor_dict['mini_batch'],
                                    max_iter=anchor_dict['max_iter']).fit(data).cluster_centers_
    elif anchor_type == 'kmeans':
        anchor = scipy.cluster.vq.kmeans2(data=data, k=m,
                                            iter=anchor_dict['max_iter'],
                                            minit='points')[0]
    w = AnchorG(data, anchor, s, sigma=None)
    et0 = time.time()
    print("* * * * * time * * * * *")
    print("Time for computing RHO: %f s" % (et0 - st0))
    print("* * * * * mid * * * * *")
    tt = et0 - st0
    w[np.eye(n, dtype=np.bool)] = 0
    w = w + delta
    w = w / w.sum(axis=1, keepdims=1)
    rho = np.sum(w, axis=0)

    return np.array(rho, np.float32), tt

def density_fkd(data, nn_dict=None):
    n = data.shape[0]
    d = data.shape[1]
    k = nn_dict['k']
    nn_type = nn_dict['type']
    delta = nn_dict['delta']
    st0 = time.time()
    if nn_type in ['kd_tree', 'ball_tree']:
        neighbor = NearestNeighbors(n_neighbors=k+1, algorithm=nn_type).fit(data).kneighbors_graph(data).toarray()
        wk = np.zeros((n, n), dtype=np.float32)
        ss = np.where(neighbor != 0)
        lens = ss[0].size
        for i in range(lens):
            wk[ss[0][i], ss[1][i]] = exp(sqeuclidean(data[ss[0][i], :], data[ss[1][i], :]) / -0.5)
    et0 = time.time()
    print("* * * * * time * * * * *")
    print("Time for computing RHO: %f s" % (et0 - st0))
    wk[np.eye(n, dtype=np.bool)] = 0
    wk = wk + delta
    wk = wk / wk.sum(axis=1, keepdims=1)
    rho = np.sum(wk, axis=0)
    tt = et0 - st0
    return np.array(rho, np.float32), tt

def density_naive(data, nn_dict=None):
    eps = nn_dict['eps']
    nn_type = nn_dict['type']
    st0 = time.time()
    graph = NearestNeighbors(radius=eps, algorithm=nn_type).fit(data).radius_neighbors_graph(data)
    rho = np.sum(graph, axis=0)
    rho = np.array(rho).flatten()
    et0 = time.time()
    print("* * * * * time * * * * *")
    print("Time for computing RHO: %f s" % (et0 - st0))
    tt = et0 - st0
    return np.array(rho, np.float32), tt

def density_lc(data, nn_dict=None):
    n = data.shape[0]
    eps = nn_dict['eps']
    k = nn_dict['k']
    nn_type = nn_dict['type']
    st0 = time.time()
    rho_naive, _ = density_naive(data, nn_dict)
    rho = np.zeros(n)
    kgraph = NearestNeighbors(n_neighbors=k+1, algorithm=nn_type).fit(data).kneighbors_graph(data).astype('int')
    for i in range(n):
        neighbor = kgraph[i, :].toarray().flatten().nonzero()[0].tolist()
        rho[i] = np.count_nonzero(rho_naive[neighbor] < rho_naive[i])
    et0 = time.time()
    print("* * * * * time * * * * *")
    print("Time for computing RHO: %f s" % (et0 - st0))
    tt = et0 - st0
    return np.array(rho, np.float32), tt

def density_rnn(data, nn_dict=None):
    n = data.shape[0]
    k = nn_dict['k']
    nn_type = nn_dict['type']
    st0 = time.time()
    kgraph = NearestNeighbors(n_neighbors=k + 1, algorithm=nn_type).fit(data).kneighbors_graph(data).astype('int')
    rho = np.sum(kgraph, axis=0)
    rho = np.array(rho).flatten()
    et0 = time.time()
    print("* * * * * time * * * * *")
    print("Time for computing RHO: %f s" % (et0 - st0))
    tt = et0 - st0
    return np.array(rho, np.float32), tt