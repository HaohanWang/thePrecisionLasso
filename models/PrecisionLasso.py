__author__ = 'Haohan Wang'

import numpy as np
from utility.cg import solve_cg, A_trace, norm

class PrecisionLasso():
    def __init__(self, lmbd=1., eps=10**(-6), P=None, maxIter=100, gamma=0.5):
        self.lmbd = lmbd
        self.P = P
        self.eps = eps
        self.maxIter = maxIter
        self.gamma = gamma

    def fit(self, X, y, cg=True, mu=None):
        P = X
        n, p = X.shape
        k_max = 0
        w = np.zeros(p)
        b = np.dot(X.T, y)
        if not cg:
            G = np.dot(X.T, X)

        if mu == None:
            mu = 10**(-2 -8 * np.arange(self.maxIter) / float(self.maxIter))
        cf = 0
        for it in range(self.maxIter):
            w[w<mu[it]] = mu[it]
            W = P * np.tile(w, (P.shape[0], 1))
            s1, U1 = np.linalg.eigh(np.dot(W, W.T))
            s1[np.abs(s1) < mu[it]] = mu[it]
            s1 = np.abs(s1)

            W = P * np.tile(1.0/w, (P.shape[0], 1))
            s2, U2 = np.linalg.eigh(np.dot(W, W.T))
            s2[np.abs(s2) < mu[it]] = mu[it]

            s2 = np.abs(1.0/s2)

            # objective_value = 0.5 * np.sum((np.dot(X, w) - y)**2) \
            #     + self.lmbd * (self.gamma*np.sum(np.sqrt(s1)) + (1-self.gamma*np.sum(np.sqrt(s2))))
            s1 = np.sqrt(s1)
            s2 = np.sqrt(s2)

            U1 = np.dot(P.T, U1)
            D1 = np.sum(U1 * U1 * np.tile(1. / s1, (U1.shape[0], 1)), 1)
            maxi = np.max(D1)
            mini = np.min(D1)
            D1 = (D1-mini)/(maxi-mini)

            U2 = np.dot(P.T, U2)
            D2 = np.sum(U2 * U2 * np.tile(1. / s2, (U2.shape[0], 1)), 1)

            maxi = np.max(D2)
            mini = np.min(D2)
            D2 = (D2-mini)/(maxi-mini)

            D = self.lmbd * (self.gamma*D1 + (1 - self.gamma)*D2)

            w_old = w
            if cg:
                w, k = solve_cg(A_trace, b, w, {'M':X, 'D':D}, tol=self.eps, k_max=self.maxIter)
            else:
                w = np.linalg.solve(G + np.diag(D), b)
            cf = it
            if norm(w - w_old) < self.eps:
                break
        w[w<=mu[cf]] = 0
        self.w = w
        return cf

    def predict(self, X):
        return np.dot(X, self.w)

    def setLambda(self, lmbd):
        self.lmbd = lmbd

    def setGamma(self, gamma):
        self.gamma = gamma

    def getBeta(self):
        return self.w

    def calculateGamma(self, X, rate=0.99):
        from utility.consistency import testRepresentability
        m = np.corrcoef(X.T) - np.diag(np.ones(X.shape[1]))
        c = np.max(m, 1)
        correlation = len(np.where(c>rate)[0])
        r = testRepresentability(X)
        linearDepend = len(np.where(r>0)[0])
        self.gamma = (float(correlation)/(correlation+linearDepend))
        print 'Model set Gamma automatically as:', self.gamma