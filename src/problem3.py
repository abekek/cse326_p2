# -------------------------------------------------------------------------
'''
    Problem 3: SMO training algorithm

'''
from problem1 import *
from problem2 import *

import numpy as np

import copy

class SVMModel():
    """
    The class containing information about the SVM model, including parameters, data, and hyperparameters.

    DONT CHANGE THIS DEFINITION!
    """
    def __init__(self, train_X, train_y, C, kernel_function, sigma=1):
        """
            train_X: n x m training feature matrix. n: number of features; m: number training examples.
            train_y: 1 x m labels (-1 or 1) of training data.
            C: a positive scalar
            kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
            sigma: need to be provided when Gaussian kernel is used.
        """
        # data
        self.train_X = train_X
        self.train_y = train_y
        self.n, self.m = train_X.shape

        # hyper-parameters
        self.C = C
        self.kernel_func = kernel_function
        self.sigma = sigma

        # parameters
        self.alpha = np.zeros((1, self.m))
        self.b = 0

def train(model, max_iters = 10, record_every = 1, max_passes = 1, tol=1e-6):
    """
    SMO training of SVM
    model: an SVMModel
    max_iters: how many iterations of optimization
    record_every: record intermediate dual and primal objective values and models every record_every iterations
    max_passes: each iteration can have maximally max_passes without change any alpha, used in the SMO alpha selection.
    tol: numerical tolerance (exact equality of two floating numbers may be impossible).
    :return: 4 lists (of iteration numbers, dual objectives, primal objectives, and models)
    Hint: refer to subsection 3.5 "SMO" in notes.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    #########################################
    iteration_numbers = []
    dual_objectives = []
    primal_objectives = []
    models = []

    for i in range(max_iters):
        # SMO
        n, m = model.train_X.shape
        alpha = model.alpha
        count = 0
        while True:
            count += 1
            alpha_old = copy.deepcopy(alpha)
            for j in range(n):
                i = np.random.randint(0, m)
                while i == j and count < 1000:
                    i = np.random.randint(0, n-1)
                    count += 1
                if model.kernel_func.__name__ == 'linear_kernel':
                    K = model.kernel_func(model.train_X, model.train_X)
                elif model.kernel_func.__name__ == 'Gaussian_kernel':
                    K = model.kernel_func(model.train_X, model.train_X, model.sigma)
                else:
                    raise ValueError('Unknown kernel function')
                E_i = np.dot(alpha * model.train_y, K[:, i]) + model.b - model.train_y[0, i]
                E_j = np.dot(alpha * model.train_y, K[:, j]) + model.b - model.train_y[0, j]
                alpha_i_old = alpha[0, i]
                alpha_j_old = alpha[0, j]        
                # compute L and H
                if model.train_y[0, i] != model.train_y[0, j]:
                    L = max(0, alpha[0, j] - alpha[0, i])
                    H = min(model.C, model.C + alpha[0, j] - alpha[0, i])
                else:
                    L = max(0, alpha[0, i] + alpha[0, j] - model.C)
                    H = min(model.C, alpha[0, i] + alpha[0, j])
                if L == H:
                    continue
                # compute eta
                if model.kernel_func.__name__ == 'linear_kernel':
                    K = model.kernel_func(model.train_X, model.train_X)
                elif model.kernel_func.__name__ == 'Gaussian_kernel':
                    K = model.kernel_func(model.train_X, model.train_X, model.sigma)
                else:
                    raise ValueError('Unknown kernel function')
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                # compute alpha_j
                alpha[0, j] -= model.train_y[0, j] * (model.alpha[0, i] - alpha_old[0, i]) / eta
                # clip alpha_j
                if alpha[0, j] > H:
                    alpha[0, j] = H
                elif alpha[0, j] < L:
                    alpha[0, j] = L
                # compute alpha_i
                alpha[0, i] += model.train_y[0, i] * model.train_y[0, j] * (alpha_old[0, j] - alpha[0, j])
            if np.linalg.norm(alpha - alpha_old) < tol:
                break
            if count > max_passes:
                break
        model.alpha = alpha
        # compute b
        b_tmp = model.train_y - np.dot(model.alpha * model.train_y, K)
        model.b = np.mean(b_tmp)
        # compute dual objective
        dual_objective = dual_objective_function(model.alpha, model.train_y, model.train_X, model.kernel_func, model.sigma)
        primal_objective = 0
        for i in range(m):
            for j in range(m):
                primal_objective += alpha[0, i] * alpha[0, j] * model.train_y[0, i] * model.train_y[0, j] * K[i, j]
        primal_objective /= 2
        for i in range(m):
            primal_objective -= alpha[0, i]

        # record logs
        if i % record_every == 0:
            iteration_numbers.append(i)
            dual_objectives.append(dual_objective)
            primal_objectives.append(primal_objective)
            models.append(copy.deepcopy(model))
        
    return iteration_numbers, dual_objectives, primal_objectives, models


def predict(model, test_X):
    """
    Predict the labels of test_X
    model: an SVMModel
    test_X: n x m matrix, test feature vectors
    :return: 1 x m matrix, predicted labels
    """
    #########################################
    ## INSERT YOUR CODE HERE
    #########################################
    if model.kernel_func.__name__ == 'linear_kernel':
        K = model.kernel_func(model.train_X, test_X)
    elif model.kernel_func.__name__ == 'Gaussian_kernel':
        K = model.kernel_func(model.train_X, test_X, model.sigma)
    else:
        raise ValueError('Unknown kernel function')
    y = np.dot(model.alpha * model.train_y, K) + model.b
    y[y >= 0] = 1
    y[y < 0] = -1
    return y
