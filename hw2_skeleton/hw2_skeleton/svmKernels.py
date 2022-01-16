"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    #TODO
    return ((X1.dot(X2.T)) + 1)** _polyDegree



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    result = -(pairwise(X1, X2))/(2 * (_gaussSigma ** 2))
    return np.exp(result)

def pairwise(X1, X2):
    n = len(X1)
    d = len(X2)

    result = np.zeros([n, d])
    for i in range(0, n):
        for j in range(0, d):
            result[i, j] = np.linalg.norm(X1[i] - X2[j]) **2
    return result

def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return #TODO (CIS 519 ONLY)

