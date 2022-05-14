import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.io
from utils import pdist2

def AnchorG(X, anchors, s, sigma):
    X = X.astype('float32')
    anchors = anchors.astype('float32')
    n = X.shape[0]
    m = anchors.shape[0]

    sqdist = pdist2(X, anchors, 'sqeuclidean')

    del anchors
    del X

    val = np.zeros((n, s), dtype=np.float32)
    pos = np.zeros((n, s), dtype=np.int)
    for i in range(s):
        pos[:, i] = np.argmin(sqdist, 1)
        val[:, i] = sqdist[np.arange(len(sqdist)), pos[:, i]]
        sqdist[np.arange(n), pos[:, i]] = float('inf')
    
    del sqdist

    if sigma is None:
        dist = np.sqrt(val[:, s - 1])
        sigma = np.mean(dist) / np.sqrt(2)

    # Calculate formula (2) from the paper. This calculation differs from the reference matlab.
    # In the matlab, the RBF kernel's exponent only has sigma^2 in the denominator. Here, 2 * sigma^2.
    # This is accounted for when auto-calculating sigma above by dividing by sqrt(2).
    # Work in log space and then exponentiate, to avoid the floating point issues. For the
    # denominator, the following code avoids even more precision issues, by relying on the fact that
    # the log of the sum of exponentials, equals some constant plus the log of sum of exponentials
    # of numbers subtracted by the constant:
    #  log(sum_i(exp(x_i))) = m + log(sum_i(exp(x_i-m)))

    c = 2 * np.power(sigma, 2)  # bandwidth parameter
    exponent = -val / c  # exponent of RBF kernel
    # shift = np.amin(exponent, 1, keepdims=True)
    # denom = np.log(np.sum(np.exp(exponent - shift), 1, keepdims=True)) + shift
    # val = np.exp(exponent - denom)
    val = np.exp(exponent)

    z = scipy.sparse.lil_matrix((n, m), dtype='float32')
    for i in range(s):
        z[np.arange(n), pos[:, i]] = val[:, i]

    # del val
    # del pos

    z = z.tocsr()
    # # mid = np.diag(np.power(np.asarray(z.sum(0)).ravel(), -1))
    # # w = np.asarray(z.dot(mid).dot(z.T.todense()))
    w = z.dot(z.T)
    w = np.asarray(w.todense(), dtype=np.float32)

    return w