__author__ = 'Haohan Wang'

import numpy as np

def testRepresentability(data):
    l = []
    [m, n] = data.shape
    for i in range(n):
        X1 = data[:, i]
        X2 = np.delete(data, i, 1)
        C11 = np.dot(X1.T, X1) * 1.0 / n
        C21 = np.dot(X2.T, X1) * 1.0 / n

        ii = 1.0 / C11
        r = np.abs(np.dot(C21, ii))
        c = len(np.where(r >= 1)[0])
        l.append(c)
    return np.array(l)

def calculateGammaAll(X, rate=0.99):
    m = np.corrcoef(X.T) - np.diag(np.ones(X.shape[1]))
    c = np.max(m, 1)
    correlation = len(np.where(c>rate)[0])
    r = testRepresentability(X)
    linearDepend = len(np.where(r>0)[0])
    return correlation, linearDepend, (float(correlation)/(correlation+linearDepend)), correlation/float(X.shape[1]), linearDepend/float(X.shape[1])
