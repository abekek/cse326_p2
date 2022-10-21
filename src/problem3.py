# -------------------------------------------------------------------------
'''
    Problem 3: SMO training algorithm

'''
from random import random
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

def get_rnd_int(a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = np.random.randint(a,b)
            cnt=cnt+1
        return i

def calculate_E(model, i):
    if model.kernel_func.__name__ == 'linear_kernel':
        K = model.kernel_func(model.train_X, model.train_X)
    elif model.kernel_func.__name__ == 'Gaussian_kernel':
        K = model.kernel_func(model.train_X, model.train_X, model.sigma)
    else:
        raise ValueError('Unknown kernel function')
    y = np.dot(model.alpha * model.train_y, K[:, i]) + model.b
    return y - model.train_y[0, i]


def take_step(model, i1, i2, Ej, tol):
    if i1 == i2:
        return False

    Ei = calculate_E(model, i1)
            
    if model.train_y[0, i1] != model.train_y[0, i2]:
        L = max(0, model.alpha[0, i2] - model.alpha[0, i1])
        H = min(model.C, model.C + model.alpha[0, i2] - model.alpha[0, i1])
    else:
        L = max(0, model.alpha[0, i2] + model.alpha[0, i1] - model.C)
        H = min(model.C, model.alpha[0, i2] + model.alpha[0, i1])
    
    if L == H:
        return False

    k11, k12, k22 = 0, 0, 0
    
    if model.kernel_func.__name__ == 'linear_kernel':
        k11 = model.kernel_func(model.train_X, model.train_X)[i1, i1]
        k12 = model.kernel_func(model.train_X, model.train_X)[i1, i2]
        k22 = model.kernel_func(model.train_X, model.train_X)[i2, i2]
    elif model.kernel_func.__name__ == 'Gaussian_kernel':
        k11 = model.kernel_func(model.train_X, model.train_X, model.sigma)[i1, i1]
        k12 = model.kernel_func(model.train_X, model.train_X, model.sigma)[i1, i2]
        k22 = model.kernel_func(model.train_X, model.train_X, model.sigma)[i2, i2]
    
    eta = 2.0 * k12 - k11 - k22

    alpha_i_old = model.alpha[0, i1]
    alpha_j_old = model.alpha[0, i2]
    
    if eta < 0:
        a2 = alpha_j_old - model.train_y[0, i2] * (Ei - Ej) / eta
        if a2 < L:
            a2 = L
        elif a2 > H:
            a2 = H
    else:
        f1 = model.train_y[0, i1] * (Ei + model.b) - model.alpha[0, i1] * k11 - model.train_y[0, i2] * model.alpha[0, i2] * k12
        f2 = model.train_y[0, i2] * (Ej + model.b) - model.alpha[0, i2] * k22 - model.train_y[0, i1] * model.alpha[0, i1] * k12
        L1 = model.alpha[0, i1] + model.alpha[0, i2] - H
        H1 = model.alpha[0, i1] + model.alpha[0, i2] - L
        Lobj = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k11 + 0.5 * L * L * k22 + L * L1 * k12
        Hobj = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k11 + 0.5 * H * H * k22 + H * H1 * k12
        if Lobj < Hobj - tol:
            a2 = H
        elif Lobj > Hobj + tol:
            a2 = L
        else:
            a2 = model.alpha[0, i2]
        print('test')
    
    if a2 < 1e-8:
        a2 = 0
    elif a2 > model.C - 1e-8:
        a2 = model.C
    
    if abs(a2 - alpha_j_old) < tol * (a2 + alpha_j_old + tol):
        return False
    
    a1 = alpha_i_old + model.train_y[0, i1] * model.train_y[0, i2] * (alpha_j_old - a2)
    b1 = model.b - Ei - model.train_y[0, i1] * (a1 - alpha_i_old) * k11 - model.train_y[0, i2] * (a2 - alpha_j_old) * k12
    b2 = model.b - Ej - model.train_y[0, i1] * (a1 - alpha_i_old) * k12 - model.train_y[0, i2] * (a2 - alpha_j_old) * k22
    
    if 0 < a1 < model.C:
        model.b = b1
    elif 0 < a2 < model.C:
        model.b = b2
    else:
        model.b = (b1 + b2) / 2.0
    
    model.alpha[0, i1] = a1
    model.alpha[0, i2] = a2
    # print('i: ', i1)
    return True

def examine_example(model, i2, tol, max_passes):
    y2 = model.train_y[0, i2]
    alpha2 = model.alpha[0, i2]
    E2 = calculate_E(model, i2)
    r2 = E2 * y2
    passes = 0
    if (r2 < -tol and alpha2 < model.C) or (r2 > tol and alpha2 > 0):
        # if number of non-zero and non-C alpha's > 1
        # if len(model.alpha[model.alpha != 0]) + len(model.alpha[model.alpha != model.C]) > 1:
        if len(model.alpha[model.alpha != 0]) > 1 and len(model.alpha[model.alpha != model.C]) > 1:
            # i1 = get_rnd_int(0, model.m, i2)
            i1 = np.random.randint(0, model.m - 1)
            if take_step(model, i1, i2, E2, tol):
                return 1
        # loop over all non-zero and non-C alpha, starting at a random point
        non_zero = np.where(model.alpha != 0)[1]
        # print(non_zero)
        non_C = np.where(model.alpha != model.C)[1]
        rnd = np.random.permutation(non_zero)
        for i1 in rnd:
            if take_step(model, i1, i2, E2, tol):
                return 1
        rnd = np.random.permutation(non_C)
        for i1 in rnd:
            if take_step(model, i1, i2, E2, tol):
                return 1
        # loop over all possible i1, starting at a random point
        rnd = np.random.permutation(model.m)
        for i1 in rnd:
            if take_step(model, i1, i2, E2, tol):
                return 1
    return 0
    

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

    for t in range(max_iters):    
        num_changed_alphas = 0
        examineAll = True
        print('iteration: ', t)

        while(num_changed_alphas > 0 or examineAll):
            num_changed_alphas = 0
            passes = 0
            if examineAll:
                for i in range(model.train_X.shape[1]):
                    new_num_alpha = examine_example(model, i, tol, max_passes)
                    num_changed_alphas += new_num_alpha
                    # if new_num_alpha == 0:
                    #     passes += 1
                    # if passes > max_passes:
                    #     break
            else:
                for i in range(model.train_X.shape[1]):
                    if model.alpha[0, i] > 0 and model.alpha[0, i] < model.C:
                        new_num_alpha = examine_example(model, i, tol, max_passes)
                        num_changed_alphas += new_num_alpha
                        # if new_num_alpha == 0:
                        #     passes += 1
                        # if passes > max_passes:
                        #     break
            if examineAll:
                examineAll = False
            elif num_changed_alphas == 0:
                examineAll = True

        dual_objective = dual_objective_function(model.alpha, model.train_y, model.train_X, model.kernel_func, model.C)
        primal_objective = primal_objective_function(model.alpha, model.train_y, model.train_X, model.b, model.C, model.kernel_func, model.sigma)

        # record logs
        if t % record_every == 0:
            iteration_numbers.append(t)
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
