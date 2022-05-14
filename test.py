import numpy as np 
from dbscan import DBSCAN
from utils import print_me, myplot
import pandas as pd
import sklearn.datasets as datasets
import scipy.io as scio
from sklearn import preprocessing
from rho import density_akd, density_fkd, density_naive, density_lc, density_rnn
import matplotlib.pyplot as plt


filename_list = ['gen1', 'gen2', 'gen3', 'test',
                 'd500', 'd1000', 'd1500', 'd2000', 'd2500',
                 'jain', 'flame', 'aggregation', 'spiral']
filename = filename_list[2]
doscale = 'no'  # no, minmax, standard, robust
rho_dict = ['naive', 'lc', 'akd', 'fkd', 'rnn']
dm = rho_dict[2]
nn_dict = {
    'type': 'kd_tree',  # ball_tree, kd_tree
    'k': 10,
    'eps': 6.5,
    'delta': 0.0001
}
anchor_dict = {
    'type': 'kmeans',  # random, kmeans, kmeanspp, mbkmeans, uniform, leader
    'm_ratio': 0.1,
    's': 10,
    'plotline': False,
    'max_iter': 10,
    'mini_batch': 20,
    'delta': 0.0001
}
threshold = -1.2
onlydensity = 'on'  # on, off


if filename in ['spiral', 'jain', 'flame', 'aggregation']:
    data = pd.read_csv(r"data/%s.txt" % filename, sep="\t", header=None)
    label_true = list(np.array(data.iloc[:, -1]).astype(dtype=int))
    data.drop(data.columns[len(data.columns) - 1], axis=1, inplace=True)
elif filename in ['olivetti', 'lfw', 'iris', 'wine', 'breast', 'digits']:
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
    data = pd.DataFrame(ds.data)
    label_true = list(ds.target)
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

if onlydensity == 'off':
    label, rho = DBSCAN(data, eps=nn_dict['eps'], threshold=threshold, dm=dm, nn_dict=nn_dict, anchor_dict=anchor_dict)
    print_me(label_true, list(label))
    print(min(label))
    print(max(label))
    # for i in range(max(label)+2):
    #     color = (label==i)
    label[1]=-1

    if data.shape[1] in [2]:
        # myplot(data, label_true, label, rho)
        plt.scatter(x=data[:, 0], y=data[:, 1], c=label)
        plt.axis = 'off'
        plt.xticks([])
        plt.yticks([])
        plt.show()

else:
    rho = None
    if dm == 'naive':
        rho = density_naive(data, nn_dict)
    elif dm == 'lc':
        rho = density_lc(data, nn_dict)
    elif dm == 'fkd':
        rho = density_fkd(data, nn_dict)
    elif dm == 'akd':
        rho = density_akd(data, anchor_dict)
    elif dm == 'rnn':
        rho = density_rnn(data, nn_dict)

    rho = preprocessing.robust_scale(rho)
    label_true[1]=-1
    plt.scatter(x=data[:, 0], y=data[:, 1], c=label_true)
    plt.axis = 'off'
    plt.xticks([])
    plt.yticks([])
    plt.show()
