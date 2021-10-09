# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 22:21:02 2021

@author: pmspr
"""
import numpy as np
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig


# define a matrix
A = array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
print(A)
print('---------')

# calculate the mean of each column
M = mean(A.T, axis=1)
print(M)
print('----------')

# center columns by subtracting column means
C = A - M
print(C)
print(C[:,0])
V = np.var(C[:,0])
print(V)
print('----------')

# calculate covariance matrix of centered matrix
V = np.cov(C.T, bias=1)
print(V)
print('-----------')

# eigendecomposition of covariance matrix
values, vectors = eig(V)
print(vectors)
print(values)
print('-----------')

# project data
P = vectors.T.dot(C.T)
print(P.T)