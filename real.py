import numpy as np
from dbscan import DBSCAN
from utils import print_me, myplot
import pandas as pd
import sklearn.datasets as datasets
import scipy.io as scio
from sklearn import preprocessing
from rho import density_akd, density_fkd, density_naive, density_lc, density_rnn
import matplotlib.pyplot as plt


filename_list = ['iris', 'wine', 'breast', 'digits', 'diabetes',
                 'libras', 'segment', 'whitewine', 'letter',
                 'pendigits', 'optdigits', 'banknote', 'satimage',
                 'olivetti', 'lfw', 'drivface']
filename = filename_list[11]
doscale = 'no'  # no, minmax, standard, robust
rho_dict = ['naive', 'lc', 'akd', 'fkd', 'rnn']
dm = rho_dict[3]
nn_dict = {
    'type': 'kd_tree',  # ball_tree, kd_tree
    'k': 10,
    'eps': 35,
    'delta': 0.0001
}
anchor_dict = {
    'type': 'kmeanspp',  # random, kmeans, kmeanspp, mbkmeans, uniform, leader
    'm_ratio': 0.1,
    's': 3,
    'plotline': False,
    'max_iter': 10,
    'mini_batch': 20,
    'delta': 0.00001
}
threshold = 0
onlydensity = 'on'  # on, off


if filename in ['spiral', 'jain', 'flame', 'aggregation']:
    data = pd.read_csv(r"data/%s.txt" % filename, sep="\t", header=None)
    label_true = list(np.array(data.iloc[:, -1]).astype(dtype=int))
    data.drop(data.columns[len(data.columns) - 1], axis=1, inplace=True)
elif filename in ['libras', 'segment', 'whitewine', 'letter',
                  'pendigits', 'optdigits', 'banknote', 'satimage']:
    data = pd.read_csv(r"data/%s.txt" % filename, sep=" ", header=None)
    label_true = list(np.array(data.iloc[:, -1]).astype(dtype=int))
    data.drop(data.columns[len(data.columns) - 1], axis=1, inplace=True)
elif filename in ['olivetti', 'lfw', 'iris', 'wine', 'breast', 'digits', 'diabetes']:
    if filename == 'olivetti':
        ds = datasets.fetch_olivetti_faces()
    elif filename == 'lfw':
        ds = datasets.fetch_lfw_people()
    elif filename == 'iris':
        ds = datasets.load_iris()
    elif filename == 'wine':
        ds = datasets.load_wine()
    elif filename == 'breast':
        ds = datasets.load_breast_cancer()
    elif filename == 'digits':
        ds = datasets.load_digits()
    elif filename == 'diabetes':
        ds = datasets.load_diabetes()
    data = pd.DataFrame(ds.data)
    label_true = list(ds.target)
elif filename in ['drivface']:
    if filename == 'drivface':
        data = pd.DataFrame(scio.loadmat('data/%s.mat' % filename)['dtxt'])
        label_true = list(np.array(data.iloc[:, -1]).astype(dtype=int))
        data.drop(data.columns[len(data.columns) - 1], axis=1, inplace=True)
elif filename in ['gen1', 'gen2', 'gen3', 'test', 'd500', 'd1000', 'd1500', 'd2000', 'd2500']:
    data = np.loadtxt('data/'+filename+'_data.txt')
    label_true = list(np.loadtxt('data/'+filename+'_label.txt'))

if doscale == 'standard':
    data = preprocessing.scale(data).astype('float32')
elif doscale == 'minmax':
    data = preprocessing.minmax_scale(data).astype('float32')
elif doscale == 'robust':
    data = preprocessing.robust_scale(data).astype('float32')
else:
    data = np.array(data, dtype=np.float32)

data = np.array(data, dtype='float32')

print('* * * * * * * * * *')
print('Data have been loaded.')
print('* * * * * * * * * *')

# if onlydensity == 'on':
#     label, rho = DBSCAN(data, eps=nn_dict['eps'], threshold=threshold, dm=dm, nn_dict=nn_dict, anchor_dict=anchor_dict)
#
#     print_me(label_true, list(label))
#
#     if data.shape[1] in [2]:
#         myplot(data, label_true, label, rho)
# else:
#     rho = None
#     if dm == 'naive':
#         rho = density_naive(data, nn_dict)
#     elif dm == 'lc':
#         rho = density_lc(data, nn_dict)
#     elif dm == 'fkd':
#         rho = density_fkd(data, nn_dict)
#     elif dm == 'akd':
#         rho = density_akd(data, anchor_dict)
#     elif dm == 'rnn':
#         rho = density_rnn(data, nn_dict)
#
#     rho = preprocessing.robust_scale(rho)
#     # plt.scatter(x=data[:, 0], y=data[:, 1], c=rho)
