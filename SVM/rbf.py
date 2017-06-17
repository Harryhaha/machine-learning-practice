__author__ = "Harry"

from numpy import *


'''
rbf implementation
X: matrix of all dataset
A: vector of a specific data item
gamma: the parameter to be adjusted
'''

def rbf(X, A, gamma):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    for j in range(m):
        deltaRow = X[j,:] - A
        K[j] = deltaRow*deltaRow.T
    K = exp(K/(-1*gamma**2))
    return K






