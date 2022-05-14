import os

import numpy as np
from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import bcubed
from metrics import pairwise
import matplotlib.pyplot as plt


def pdist2(X, Y, metric):
    # scipy has a cdist function that works like matlab's pdist2 function.
    # For square euclidean distance it is slow for the version of scipy you have.
    # For details on its slowness, see https://github.com/scipy/scipy/issues/3251
    # In your tests, it took over 16 seconds versus less than 4 seconds for the
    # implementation below (where X has 69,000 elements and Y had 300).
    # (this has squared Euclidean distances).
    metric = metric.lower()
    if metric == 'sqeuclidean':
        X = X.astype('float32')
        Y = Y.astype('float32')
        nx = X.shape[0]
        ny = Y.shape[0]
        XX = np.tile((X ** 2).sum(1), (ny, 1)).T
        YY = np.tile((Y ** 2).sum(1), (nx, 1))
        XY = X.dot(Y.T)

        # del X
        # del Y

        sqeuc = XX + YY - 2 * XY
        # Make negatives equal to zero. This arises due to floating point
        # precision issues. Negatives will be very close to zero (IIRC around
        # -1e-10 or maybe even closer to zero). Any better fix? you exhibited the
        # floating point issue on two machines using the same code and data,
        # but not on a third. the inconsistent occurrence of the issue could
        # possibly be due to differences in numpy/blas versions across machines.
        return np.clip(sqeuc, 0, np.inf)
    elif metric == 'hamming':
        # scipy cdist supports hamming distance, but is twice as slow as yours
        # (even before multiplying by dim, and casting as int), possibly because
        # it supports non-booleans, but I'm not sure...
        # Looping over data points in X and Y, and calculating hamming distance
        # to put in a hamdis matrix is too slow. This vectorized solution works
        # faster.
        hashbits = X.shape[1]
        # Use high bitwidth int to prevent overflow (i.e., as opposed to int8
        # which could result in overflow when hashbits >= 64).
        X_int = (2 * X.astype('int')) - 1
        Y_int = (2 * Y.astype('int')) - 1
        hamdis = hashbits - ((hashbits + X_int.dot(Y_int.T)) / 2)
        return hamdis
    else:
        valerr = 'Unsupported Metric: %s' % (metric,)
        raise ValueError(valerr)

def print_me(label_true, label_pred):
    print("* * * * * result * * * * *")
    n = len(label_true)
    print("RI: %f" % rand_score(label_true, label_pred))
    print("ARI: %f" % adjusted_rand_score(label_true, label_pred))
    # print("AMI: %f" % adjusted_mutual_info_score(label_true, label_pred))
    print("NMI: %f" % normalized_mutual_info_score(label_true, label_pred))
    pairwise_p, pairwise_r, pairwise_fscore = pairwise(np.array(label_true), np.array(label_pred), sparse=True)
    print("Pairwise_Precision: %f" % pairwise_p)
    print("Pairwise_Recall: %f" % pairwise_r)
    print("Pairwise_Fscore: %f" % pairwise_fscore)
    set_pred = list()
    set_true = list()
    for i in range(n):
        set_pred.append(set([label_pred[i]]))
        set_true.append(set([label_true[i]]))
    dict_pred = dict(zip(list(range(len(label_pred))), set_pred))
    dict_true = dict(zip(list(range(len(label_true))), set_true))
    # bcubed_p = bcubed.precision(dict_pred, dict_true)
    # bcubed_r = bcubed.recall(dict_pred, dict_true)
    # print("Bcubed_Precision: %f" % bcubed_p)
    # print("Bcubed_Recall: %f" % bcubed_r)
    # print("Bcubed_Fscore: %f" % bcubed.fscore(bcubed_p, bcubed_r))

def record_me(label_true, label_pred):
    print("* * * * * result * * * * *")
    n = len(label_true)
    print("RI: %f" % rand_score(label_true, label_pred))
    ari = adjusted_rand_score(label_true, label_pred)
    print("ARI: %f" % ari)
    # print("AMI: %f" % adjusted_mutual_info_score(label_true, label_pred))
    nmi = normalized_mutual_info_score(label_true, label_pred)
    print("NMI: %f" % nmi)
    pairwise_p, pairwise_r, pairwise_fscore = pairwise(np.array(label_true), np.array(label_pred), sparse=True)
    print("Pairwise_Precision: %f" % pairwise_p)
    print("Pairwise_Recall: %f" % pairwise_r)
    print("Pairwise_Fscore: %f" % pairwise_fscore)
    set_pred = list()
    set_true = list()
    for i in range(n):
        set_pred.append(set([label_pred[i]]))
        set_true.append(set([label_true[i]]))
    dict_pred = dict(zip(list(range(len(label_pred))), set_pred))
    dict_true = dict(zip(list(range(len(label_true))), set_true))
    # bcubed_p = bcubed.precision(dict_pred, dict_true)
    # bcubed_r = bcubed.recall(dict_pred, dict_true)
    # print("Bcubed_Precision: %f" % bcubed_p)
    # print("Bcubed_Recall: %f" % bcubed_r)
    # print("Bcubed_Fscore: %f" % bcubed.fscore(bcubed_p, bcubed_r))
    return ari, nmi, pairwise_fscore

def myplot(data, label_true, label, rho):
    d = data.shape[1]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.scatter(x=data[:, 0], y=data[:, 1], c=label_true)
    ax2.scatter(x=data[:, 0], y=data[:, 1], c=label)
    ax3.scatter(x=data[:, 0], y=data[:, 1], c=rho)
    plt.show()
