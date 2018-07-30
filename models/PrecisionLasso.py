__author__ = 'Haohan Wang'

import numpy as np
from utility.cg import solve_cg, A_trace, norm, logisticRegressionGradientSolver

class PrecisionLasso():
    def __init__(self, lmbd=1., eps=10**(-6), P=None, maxIter=100, gamma=0.5, lr=1e-6, tol=1e-5, mu=1e-2, logistic=False):
        self.lmbd = lmbd
        self.P = P
        self.eps = eps
        self.maxIter = maxIter
        self.gamma = gamma
        self.logistic = logistic
        self.lr = lr
        self.tol = tol
        self.mu = mu

    def setLogisticFlag(self, logistic):
        self.logistic = logistic

    def setLearningRate(self, lr):
        self.lr = lr

    def fit(self, X, y, cg=True):
        X0 = np.ones(len(y)).reshape(len(y), 1)
        X = np.hstack([X, X0])
        P = X
        n, p = X.shape
        k_max = 0
        w = np.zeros(p)
        b = np.dot(X.T, y)
        G = np.dot(X.T, X)

        # if mu == None:
        muList = 10**(2 -8 * np.arange(self.maxIter) / float(self.maxIter))
        cf = 0
        for it in range(self.maxIter):
            self.mu = muList[it]
            w = np.nan_to_num(w)
            w[np.abs(w)<self.mu] = self.mu
            W = P * np.tile(w, (P.shape[0], 1))
            s1, U1 = np.linalg.eigh(np.dot(W, W.T))
            s1[np.abs(s1) < self.mu] = self.mu
            s1 = np.abs(s1)

            W = P * np.tile(1.0/w, (P.shape[0], 1))
            s2, U2 = np.linalg.eigh(np.dot(W, W.T))
            s2[np.abs(s2) < self.mu] = self.mu

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
            if self.logistic:
                w = logisticRegressionGradientSolver(w=w, X=X, y=y, D=D, lr=self.lr, tol=self.tol, maxIter=self.maxIter, quiet=True)
                if w is None:
                    self.w = np.zeros(p)
                    return None
            else:
                if cg:
                    w, k = solve_cg(A_trace, b, w, {'M':X, 'D':D}, tol=self.eps, k_max=self.maxIter)
                else:
                    w = np.linalg.solve(G + np.diag(D), b)
            cf = it
            if norm(w - w_old) < self.eps:
                break
        w = np.nan_to_num(w)
        w[np.abs(w)<self.mu] = 0
        self.w = w
        return cf

    def predict(self, X):
        X0 = np.ones(X.shape[0]).reshape(X.shape[0], 1)
        X = np.hstack([X, X0])
        if not self.logistic:
            return np.dot(X, self.w)
        else:
            t = 1. / (1 + np.exp(-np.dot(X, self.w)))
            y = np.zeros_like(t)
            y[t>0.5] = 1
            return t

    def setLambda(self, lmbd):
        self.lmbd = lmbd

    def setGamma(self, gamma):
        self.gamma = gamma

    def getBeta(self):
        self.w = self.w.reshape(self.w.shape[0])
        return self.w[:-1]

    def testRepresentability(self, data):
        l = []
        [m, n] = data.shape
        for i in range(n):
            X1 = data[:, i]
            X2 = np.delete(data, i, 1)
            C11 = np.dot(X1.T, X1) * 1.0 / n
            C21 = np.dot(X2.T, X1) * 1.0 / n

            ii = 1.0 / C11
            r = np.abs(np.dot(C21, ii))
            c = len(np.where(r >= 1+1e-5)[0])
            l.append(c)
        return np.array(l)

    def calculateGamma(self, X, rate=0.85, sample=True):
        if sample:
            np.random.seed(0)
            idx = np.random.choice(X.shape[1], X.shape[0], replace=False)
            tmpX = X[:, idx]
        else:
            tmpX = X

        mc = np.abs(np.corrcoef(tmpX.T))
        mc[np.where(np.isnan(mc))] = 1
        m = mc - np.diag(np.ones(tmpX.shape[1]))
        c = np.nanmax(m, 1)
        correlation = len(np.where(c>rate)[0])
        r = self.testRepresentability(tmpX)
        linearDepend = len(np.where(r>0)[0])
        if correlation+linearDepend == 0:
            self.gamma = 0.5
        else:
            self.gamma = (float(correlation)/(correlation+linearDepend))
