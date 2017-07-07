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
