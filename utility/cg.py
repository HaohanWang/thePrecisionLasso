# Copyright (c) 2012, Edouard Grave.

# This code is distributed under the BSD license.
# See LICENSE.txt for more information.

import numpy as np

def solve_cg(A, b, x, args, tol=10**(-8), k_max=None):
    if k_max == None:
        k_max = x.shape[0]
    k = 0
    r = b - A(x, **args)
    rho_0 = np.dot(r, r)
    rho_1 = rho_0
    while (rho_1 > tol) and (k < k_max):
        k += 1
        if k == 1:
            p = r
        else:
            beta = rho_1 / rho_0
            p = r + beta * p
        w = A(p, **args)
        alpha = rho_1 / np.dot(p, w)
        x = x + alpha * p
        r = r - alpha * w
        rho_0 = rho_1
        rho_1 = np.dot(r, r)
    return x, k

def A_trace(x, M, D):
    return np.dot(M.T, np.dot(M, x)) + D * x

def norm(w):
    return np.sqrt(np.sum(w**2))

def logisticRegressionSolverCost(t):
    return np.linalg.norm(t)

def logisticGradient(X, D, Xi, diff):
    return -np.dot(X.transpose(), diff) - np.multiply(D, np.dot(diff, Xi.T))

def stopCheck(prev, new, pg, X, y):
    if np.linalg.norm(y - np.dot(X, new)) <= \
                            np.linalg.norm(y - np.dot(X, new)) + np.sum(np.dot(pg.transpose(), (new - prev))):
        return False
    else:
        return True

def logisticRegressionGradientSolver(w, X, y, D, lr, tol, maxIter):
    resi_prev = np.inf
    tmp = 1/(1+np.exp(-np.dot(X, w)))

    xi = np.dot(X.T, np.linalg.pinv(np.dot(X, X.T)))

    Dxw = np.dot(np.multiply(D, w), xi)
    diff = y - tmp - Dxw
    resi = logisticRegressionSolverCost(diff)

    step = 0
    while resi_prev - resi > tol and step < maxIter:
        keepRunning = True
        resi_prev = resi
        runningStep = 0
        while keepRunning and runningStep < 10:
            runningStep += 1
            prev_w = w
            grad = logisticGradient(X, D, xi, diff)
            w = w - grad * lr
            tmp = 1/(1+np.exp(-np.dot(X, w)))
            Dxw = np.dot(np.multiply(D, w), xi)
            diff = y - tmp - Dxw
            keepRunning = stopCheck(prev_w, w, grad, X, y)
            # keepRunning = False
            if keepRunning:
                lr = 0.5 * lr

        step += 1
        resi = logisticRegressionSolverCost(diff)
        if resi > resi_prev:
            return w
    return w
