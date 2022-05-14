import numpy
import numpy as np
from dbscan import DBSCAN
from utils import print_me, myplot, record_me
import pandas as pd
import sklearn.datasets as datasets
import scipy.io as scio
from sklearn import preprocessing
from rho import density_akd, density_fkd, density_naive, density_lc, density_rnn
import matplotlib.pyplot as plt


rho_dict = ['naive', 'lc', 'rnn', 'fkd', 'akd']
data_list = ['gen2', 'gen3', 'iris', 'banknote', 'segment', 'pendigits']

th_pack = [[-5, 0, 0, 0.2, 0.1],
           [-0.9, 0.9, 0.5, 0.5, 0.1],
           [0.15, 0.15, 0.15, 0.15, 0.15],
           [0.15, 0.15, 0.15, 0.15, 0.15],
           [0, 0.15, 0.15, 0.15, 0.15],
           [0.02, 0.2, 0.2, 0.2, 0.2]]
k_pack = [[10, 10, 10, 20, 10],
          [10, 10, 10, 10, 10],
          [5, 5, 5, 5, 5],
          [20, 20, 20, 20, 20],
          [10, 10, 10, 10, 10],
          [20, 50, 20, 20, 20]]
eps_pack = [[6, 6, 6, 6.5, 6],
            [6, 6, 6, 6, 6],
            [0.45, 0.45, 0.45, 0.45, 0.45],
            [0.12, 0.12, 0.16, 0.16, 0.16],
            [32, 32, 35, 35, 35],
            [0.39, 0.39, 0.39, 0.39, 0.39]]
s_pack = [10, 10, 2, 2, 3, 2]
delta_pack = [0.001, 0.001, 0.00001, 0.00001, 0.00001, 0.0001]
delta1_pack = [0.0001, 0.0001, 0.00001, 0.00001, 0.00001, 0.00001]
scale_pack = ['no', 'no', 'minmax', 'minmax', 'no', 'minmax']

a = len(data_list)
b = len(rho_dict)
c = 5
list_ari = np.zeros((a, b, c))
list_nmi = np.zeros((a, b, c))
list_fscore = np.zeros((a, b, c))

for i in range(a):
    for j in range(b):
        for k in range(c):
            anchor_list = dict()
            anchor_list['type'] = 'kmeans'
            anchor_list['m_ratio'] = 0.1
            anchor_list['max_iter'] = 15
            anchor_list['plotline'] = False
            nn_dict = dict()
            nn_dict['type'] = 'kd_tree'
            nn_dict['delta'] = delta1_pack[i]
            filename = data_list[i]
            dm = rho_dict[j]
            threshold = th_pack[i][j]
            doscale = scale_pack[i]
            anchor_list['s'] = s_pack[i]
            anchor_list['delta'] = delta_pack[i]
            nn_dict['k'] = k_pack[i][j]
            nn_dict['eps'] = eps_pack[i][j]
            if filename=='booknote':
                anchor_list['type'] = 'kmeanspp'

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
            label, rho = DBSCAN(data, eps=nn_dict['eps'], threshold=threshold, dm=dm, nn_dict=nn_dict, anchor_dict=anchor_list)
            ari, nmi, pairwise_fscore = record_me(label_true, list(label))
            list_ari[i, j, k] = ari
            list_nmi[i, j, k] = nmi
            list_fscore[i, j, k] = pairwise_fscore

numpy.save('result/ari', list_ari)
numpy.save('result/nmi', list_nmi)
numpy.save('result/fscore', list_fscore)
# list_ari = numpy.load('ari.npy')
ari_mean = list_ari.mean(axis=2)
nmi_mean = list_nmi.mean(axis=2)
fscore_mean = list_fscore.mean(axis=2)
ari_std = list_ari.std(axis=2)
nmi_std = list_nmi.std(axis=2)
fscore_std = list_fscore.std(axis=2)
numpy.save('result/ari_mean', ari_mean)
numpy.save('result/nmi_mean', nmi_mean)
numpy.save('result/fscore_mean', fscore_mean)
numpy.save('result/ari_std', ari_std)
numpy.save('result/nmi_std', nmi_std)
numpy.save('result/fscore_std', fscore_std)
