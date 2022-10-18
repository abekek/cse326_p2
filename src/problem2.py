# -------------------------------------------------------------------------
'''
    Problem 2: Compute the objective function and decision function of dual SVM.

'''
from problem1 import *

import numpy as np

# -------------------------------------------------------------------------
def dual_objective_function(alpha, train_y, train_X, kernel_function, sigma):
    """
    Compute the dual objective function value.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix. n: number of features; m: number training examples.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.
    :return: a scalar representing the dual objective function value at alpha
    Hint: refer to the objective function of Eq. (47).
          You can try to call kernel_function.__name__ to figure out which kernel are used.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    #########################################
    m = train_X.shape[1]
    if kernel_function.__name__ == 'linear_kernel':
        K = kernel_function(train_X, train_X)
    elif kernel_function.__name__ == 'Gaussian_kernel':
        K = kernel_function(train_X, train_X, sigma)
    else:
        raise ValueError('Unknown kernel function')
    summation = 0
    for i in range(m):
        for j in range(m):
            summation += alpha[0, i] * alpha[0, j] * train_y[0, i] * train_y[0, j] * K[i, j]
    return np.sum(alpha) - 1/2 * summation


# -------------------------------------------------------------------------
def primal_objective_function(alpha, train_y, train_X, b, C, kernel_function, sigma):
    """
    Compute the primal objective function value.
    When with linear kernel:
        The primal parameter w is recovered from the dual variable alpha.
    When with Gaussian kernel:
        Can't recover the primal parameter and kernel trick needs to be used to compute the primal objective function.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    b: bias term
    C: regularization parameter of soft-SVM
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: a scalar representing the primal objective function value at alpha
    Hint: you need to use kernel trick when come to Gaussian kernel. Refer to the derivation of the dual objective function Eq. (47) to check how to find
            1/2 ||w||^2 and the decision_function with kernel trick.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    #########################################
    if kernel_function.__name__ == 'linear_kernel':
        K = kernel_function(train_X, train_X)
        w = np.dot(alpha * train_y, train_X.T)
        return 1/2 * np.dot(w, w.T) + C * np.sum(np.maximum(0, 1 - train_y * (np.dot(w, train_X) + b)))
    elif kernel_function.__name__ == 'Gaussian_kernel':
        K = kernel_function(train_X, train_X, sigma)
        summation = 0
        for i in range(train_X.shape[1]):
            summation += alpha[0, i] * train_y[0, i] * np.sum(alpha * train_y * K[:, i])
        return 1/2 * summation + C * np.sum(np.maximum(0, 1 - train_y * (np.dot(alpha * train_y, K) + b)))
    else:
        raise ValueError('Unknown kernel function')



def decision_function(alpha, train_y, train_X, b, kernel_function, sigma, test_X):
    """
    Compute the linear function <w, x> + b on examples in test_X, using the current SVM.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    test_X: n x m2 test feature matrix.
    b: scalar, the bias term in SVM <w, x> + b.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: 1 x m2 vector <w, x> + b
    """
    #########################################
    ## INSERT YOUR CODE HERE
    #########################################
    if kernel_function.__name__ == 'linear_kernel':
        w = np.dot(alpha * train_y, train_X.T)
        return np.dot(w, test_X) + b
    elif kernel_function.__name__ == 'Gaussian_kernel':
        K = kernel_function(train_X, test_X, sigma)
        return np.dot(alpha * train_y, K) + b
    else:
        raise ValueError('Unknown kernel function')
