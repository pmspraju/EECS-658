#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:55:15 2021

@author: ritikanair
"""

import sys
# scipy
import scipy
# numpy
import numpy as np
# pandas
import pandas as pd

# scikit-learn
import sklearn

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 6\EECS 658\Data\iris.csv'
dataset = pd.read_csv(path) #to read the data from iris.csv into the variable 'dataset'

X = dataset
y = dataset.pop('class')


# Encode the label
from sklearn import preprocessing
le = preprocessing.LabelEncoder() 
le.fit(y) #figure out the y column entries as  0,1,2

#print('Classes of the label:')
#print(le.classes_)

y = list(le.transform(y)) #assign labels

#Split Data into 2 Folds for Training and Test
#2-fold cross-validation
from sklearn.model_selection import train_test_split
X_trainFold1, X_testFold1, y_trainFold1, y_testFold1 = train_test_split(X, y, test_size=0.50, random_state=1) 
X_trainFold2 = X_testFold1
X_testFold2 = X_trainFold1
y_trainFold2 = y_testFold1
y_testFold2 = y_trainFold1

#Concatenating the test results from the 2 folds to get a test set of 150 samples
concat_actual = np.concatenate((y_testFold1,y_testFold2))
#oncat_actual = le.inverse_transform(concat_actual)

#Part 1:SVM

from sklearn.svm import LinearSVC

model_SVM = LinearSVC(max_iter=10000)
model_SVM.fit(X_trainFold1,y_trainFold1)
pred_foldSVM1 = model_SVM.predict(X_testFold1)
model_SVM.fit(X_trainFold2,y_trainFold2)
pred_foldSVM2 = model_SVM.predict(X_testFold2) 


concat_predSVM = np.concatenate((pred_foldSVM1,pred_foldSVM2))
#concat_predSVM= le.inverse_transform(concat_predSVM)
#print(concat_predSVM)

#Part 2: PCA Feature Transformation
    
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

#print(X)

M=mean(X.T,axis=1) #axis = o by default
#print(M)
X_cent = X-M
#print("X_cent=",X_cent) #X_centered matrix
X_cov= cov(X_cent.T) #covariance matrix
#print(X_cov)
# eigendecomposition of covariance matrix
V, W = eig(X_cov) #V is eigenvalue matrix, W is eigenvector matrix

print("Eigen Vectors:")
print(W)
print('-----------')
print("Eigen Values:")
print(V)

# project data
#Z = X_cent.dot(W.T)
Z=W.T.dot(X_cent.T)
print("Z =",Z.T)

L1 = V[0] #+ V[1]
SumL = np.sum(V)
pov = L1/SumL
print('Proportion of Variance (PoV)',pov)

#S = V[0]+V[1]+ V[2] + V[3]
#s=V[0]
#for i in range(4):
#    s= s
#    PoV = s/S
#    if PoV > 0.9:
#        
#        break

iris_subset = Z[0]
iris_subset_df = pd.DataFrame(iris_subset, columns=['sepal-length-Pca'])

Z_trainFold1, Z_testFold1, y_trainFold1, y_testFold1 = train_test_split(iris_subset_df, y, test_size=0.50, random_state=1) 
Z_trainFold2 = Z_testFold1
Z_testFold2 = Z_trainFold1
y_trainFold2 = y_testFold1
y_testFold2 = y_trainFold1

model_SVM = LinearSVC(max_iter=10000)
model_SVM.fit(Z_trainFold1,y_trainFold1)
pred_foldSVM1 = model_SVM.predict(Z_testFold1)
model_SVM.fit(Z_trainFold2,y_trainFold2)
pred_foldSVM2 = model_SVM.predict(Z_testFold2) 


concat_predPCA = np.concatenate((pred_foldSVM1,pred_foldSVM2))
#concat_predSVM= le.inverse_transform(concat_predSVM)
#print(concat_predSVM)


from sklearn.metrics import accuracy_score, confusion_matrix

methods = [concat_predSVM, concat_predPCA]
methodnames = ['Part 1: Linear SVC:', 'Part 2: PCA algorithm:']
for i in range(len(methods)):
    print('\n', methodnames[i], sep = '') #sep does not create space in the start of the line
    print("\nAccuracy score = ",accuracy_score(concat_actual,methods[i]) )
    print("\nConfusion matrix = \n", confusion_matrix(concat_actual,methods[i]) )
    print("\nNumber of mislabeled classes out of a total %d points : %d" % (len(concat_actual), (concat_actual != methods[i]).sum())) # %d is a placeholder for decimal nubmers
    if i ==0:
        print("\nFeatures:",list(X.columns)) 
    if i==1:
         print("\nFeatures:",list(iris_subset_df.columns)) 
         print("Eigenvector matrix:", W)
         print("Eigenvalues:", V)         

#compare with pca sklearn
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
X_pca = pca.transform(X)

print("Eigen Vectors:")
print(pca.components_)
print('-----------')

print("Eigen Values:")
print(pca.explained_variance_)
print('-----------')

#print("Projected Data")
#print(X_pca)

iris_subset = X_pca.T[0]
iris_subset_df = pd.DataFrame(iris_subset, columns=['sepal-length-Pca'])

Z_trainFold1, Z_testFold1, y_trainFold1, y_testFold1 = train_test_split(iris_subset_df, y, test_size=0.50, random_state=1) 
Z_trainFold2 = Z_testFold1
Z_testFold2 = Z_trainFold1
y_trainFold2 = y_testFold1
y_testFold2 = y_trainFold1

model_SVM = LinearSVC(max_iter=10000)
model_SVM.fit(Z_trainFold1,y_trainFold1)
pred_foldSVM1 = model_SVM.predict(Z_testFold1)
model_SVM.fit(Z_trainFold2,y_trainFold2)
pred_foldSVM2 = model_SVM.predict(Z_testFold2) 

concat_predPCA = np.concatenate((pred_foldSVM1,pred_foldSVM2))
print("\nAccuracy score = ",accuracy_score(concat_actual,concat_predPCA ))
print("\nConfusion matrix = \n", confusion_matrix(concat_actual,concat_predPCA) )