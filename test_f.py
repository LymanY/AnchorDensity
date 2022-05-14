import numpy as np
from dbscan import DBSCAN
from utils import print_me, myplot
import pandas as pd
import sklearn.datasets as datasets
import scipy.io as scio
from sklearn import preprocessing
from faissrho import density_akd, density_fkd, density_naive, density_lc, density_rnn
from faissrho import faiss_lc, faiss_fkd, faiss_rnn, faiss_naive
import matplotlib.pyplot as plt


nn_dict = {
    'type': 'ball_tree',  # ball_tree, kd_tree
    'k': 10,
    'eps': 6,
    'delta': 0.0001
}
anchor_dict = {
    'type': 'kmeans',  # random, kmeans, kmeanspp, mbkmeans, uniform, leader
    'm_ratio': 0.1,
    's': 10,
    'plotline': False,
    'max_iter': 10,
    'mini_batch': 20,
    'delta': 0.001
}
filename_list = ['gen1', 'd500', 'd1000', 'd1500', 'd2000', 'd2500']
rho_dict = ['akd', 'rakd', 'naive', 'lc', 'fkd', 'rnn', 'fnaive', 'flc', 'ffkd', 'frnn']

filename = filename_list[2]
dm = rho_dict[6]
data = np.loadtxt('data/'+filename+'_data.txt')
data = np.array(data, dtype='float32')
rho = None
if dm == 'naive':
    rho, tt = density_naive(data, nn_dict)
elif dm == 'lc':
    rho, tt = density_lc(data, nn_dict)
elif dm == 'fkd':
    rho, tt = density_fkd(data, nn_dict)
elif dm == 'rnn':
    rho, tt = density_rnn(data, nn_dict)
elif dm == 'fnaive':
    rho, tt = faiss_naive(data, nn_dict)
elif dm == 'flc':
    rho, tt = faiss_lc(data, nn_dict)
elif dm == 'ffkd':
    rho, tt = faiss_fkd(data, nn_dict)
elif dm == 'frnn':
    rho, tt = faiss_rnn(data, nn_dict)
elif dm == 'akd':
    rho, tt = density_akd(data, anchor_dict, antype='kmeans')
elif dm == 'rakd':
    rho, tt = density_akd(data, anchor_dict, antype='random')

print(tt)

