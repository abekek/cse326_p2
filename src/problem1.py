'''
    Problem 1: Implement linear and Gaussian kernels and hinge loss
'''

import numpy as np
from sklearn.metrics.pairwise  import euclidean_distances

def sparse_dot_product(id1, val1, id2, val2):
    p1, p2, dot = 0, 0, 0
    while p1 < len(id1) and p2 < len(id2):
        a1, a2 = id1[p1], id2[p2]
        if a1 == a2:
            dot += val1[p1] * val2[p2]
            p1 += 1
            p2 += 1
        elif a1 < a2:
            p1 += 1
        else:
            p2 += 1
    return dot


def linear_kernel(X1, X2):
    
    """
    Compute linear kernel between two set of feature vectors.
    The constant 1 is not appended to the x's.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    
    Note that m1 may not equal m2

    :return: if both m1 and m2 are 1, return linear kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=linear kernel evaluated on column i from X1 and column j from X2.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    #########################################

    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    m1 = X1.shape[1]
    m2 = X2.shape[1]
    K = np.zeros((m1, m2))
    if m1 == 1 and m2 == 1:
        K = np.dot(X1.T, X2)
    else:
        for i in range(m1):
            for j in range(m2):
                K[i, j] = np.dot(X1[:, i], X2[:, j])
    return K


def Gaussian_kernel(X1, X2, sigma=1):
    """
    Compute Gaussian kernel between two set of feature vectors.
    
    The constant 1 is not appended to the x's.
    
    For your convenience, please use euclidean_distances.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    sigma: Gaussian variance (called bandwidth)

    Note that m1 may not equal m2

    :return: if both m1 and m2 are 1, return Gaussian kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=Gaussian kernel evaluated on column i from X1 and column j from X2

    """
    #########################################
    ## INSERT YOUR CODE HERE
    #########################################
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    m1 = X1.shape[1]
    m2 = X2.shape[1]
    K = np.zeros((m1, m2))
    if m1 == 1 and m2 == 1:
        K = np.exp(-euclidean_distances(X1.T, X2.T) ** 2 / (2 * sigma ** 2))
    else:
        K = [[np.exp(-euclidean_distances(X1[:, i].reshape(1, -1), X2[:, j].reshape(1, -1)) ** 2 / (2 * sigma ** 2)) for j in range(m2)] for i in range(m1)]
        K = np.array(K).reshape(m1, m2)
    return K


def hinge_loss(z, y):
    """
    Compute the hinge loss on a set of training examples
    z: 1 x m vector, each entry is <w, x> + b (may be calculated using a kernel function)
    y: 1 x m label vector. Each entry is -1 or 1
    :return: 1 x m hinge losses over the m examples
    """
    #########################################
    ## INSERT YOUR CODE HERE
    #########################################
    return np.maximum(0, 1 - y * z)
