# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 05:07:04 2021

@author: pmspr
"""

# Import relevant libraries
import os
import sys
print('Python: {}'.format(sys.version))

import scipy
print('scipy: {}'.format(scipy.__version__))

import numpy as np
print('numpy: {}'.format(np.__version__))

import pandas as pd
print('pandas: {}'.format(pd.__version__))

import sklearn
print('sklearn: {}'.format(sklearn.__version__))

from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay

print("Hello World!")

#Function to load the data
def loadData(path,filename):
    try:
             files = os.listdir(path)
             for f in files:
                 if f == filename:
                     data = pd.read_csv(os.path.join(path,f))
                     return data
            
    except Exception as ex:
           print ("Exception occured in loadData-------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
# Read the file
path = r'C:\Users\pmspr\Documents\HS\MS\Sem 6\EECS 658\Data'
filename = "imbalanced iris.csv"
data = loadData(path,filename)
print(data)

print(data.groupby(['class']).size().reset_index(name='counts'))

y = data.pop('class')
X = data
y_asis = y

print('*************')
print('Part-1')
print('*************')
# Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

#Derive two folds for cross validation
X_train1 = X_train; y_train1 = y_train
X_test1 = X_test; y_test1 = y_test

X_train2 = X_test; y_train2 = y_test
X_test2 = X_train; y_test2 = y_train

print('Number of samples in fold1:{}'.format(len(X_train1)))
print('Number of samples in fold2:{}'.format(len(X_train2)))

# Model multi layer perceptron - neural network
from sklearn.neural_network import MLPClassifier
m1 = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000)

# fold1
clf1 = m1.fit(X_train1,y_train1)
y_pred1 = clf1.predict(X_test1)

# fold2
clf1 = m1.fit(X_train2,y_train2)
y_pred2 = clf1.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = y_raw #le.inverse_transform(y_raw)
y_testr = [*y_test1, *y_test2] #le.inverse_transform([*y_test1, *y_test2])

# Metrics
print("Number of mislabeled points out of a total %d points : %d" % (len(y_testr), len( [i for i, j in zip(y_testr, y_predr) if i != j] )))
        
# Accuracy
acc = accuracy_score(y_testr, y_predr)
print('Accuracy of the {:s} model: {:f}'.format('Neural network',acc))

# Confusion matrix
cmat=confusion_matrix(y_testr, y_predr)
print('Confusion Matrix of Neural network:')
print(cmat)

cmatr = np.transpose(cmat);
tpl = np.diag(cmatr);
fpl = []; fnl = []; tnl = []
for i in range(3):
    tl = [0,1,2]
    tl.remove(i)
    fpl.append(sum(cmatr[i][tl])) 
    fnl.append(sum(cmat[i][tl]))
    tnl.append(sum([sum(cmatr[j][tl]) for j in tl]))

precision   = [(tpl[i]/(tpl[i]+fpl[i]))for i in range(3)] #(tp/tp+fp)
recall      = [(tpl[i]/(tpl[i]+fnl[i]))for i in range(3)] #(tp/tp+fn)
specificity = [(tnl[i]/(tnl[i]+fpl[i]))for i in range(3)] #(tn/tn+fp)

cmetrics = pd.DataFrame(zip(precision, recall, specificity), columns=['precision','recall','specificity'], index=['Setosa','Versicolor','Virginica'])

# Class balanced accuracy
print('Class balanced accuracy of the Neural Network Model model:', sum([min(i,j) for i,j in zip(precision,recall)])/3)

# Balanced accuracy
print('Balanced accuracy of the Neural Network Model model:', np.mean([np.mean([i,j]) for i,j in zip(precision,recall)]) )

# Print sklearn balanced accuracy
from sklearn.metrics import balanced_accuracy_score
print('Sklearn balanced accuracy of the Neural Network Model model:', balanced_accuracy_score(y_testr,y_predr))

print('*************')
print('Part-2')
print('*************')
from collections import Counter
import imblearn

print('#############')
print('Random oversampling')
print('#############')

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)
print('Random oversampled dataset shape %s' % Counter(y_res))

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.5, random_state=1)

#Derive two folds for cross validation
X_train1 = X_train; y_train1 = y_train
X_test1 = X_test; y_test1 = y_test

X_train2 = X_test; y_train2 = y_test
X_test2 = X_train; y_test2 = y_train

print('Number of samples in fold1:{}'.format(len(X_train1)))
print('Number of samples in fold2:{}'.format(len(X_train2)))

# Model multi layer perceptron - neural network
from sklearn.neural_network import MLPClassifier
m2 = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000)

# fold1
clf2 = m2.fit(X_train1,y_train1)
y_pred1 = clf2.predict(X_test1)

# fold2
clf2 = m2.fit(X_train2,y_train2)
y_pred2 = clf2.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = y_raw #le.inverse_transform(y_raw)
y_testr = [*y_test1, *y_test2] #le.inverse_transform([*y_test1, *y_test2])

# Metrics
print("Number of mislabeled points out of a total %d points : %d" % (len(y_testr), len( [i for i, j in zip(y_testr, y_predr) if i != j] )))
        
# Accuracy
acc = accuracy_score(y_testr, y_predr)
print('Accuracy of the {:s} model with Random oversampling: {:f}'.format('Neural network',acc))

# Confusion matrix
cmat=confusion_matrix(y_testr, y_predr)
print('Confusion Matrix of Neural network with Random oversampling:')
print(cmat)

print('#############')
print('SMOTE oversampling')
print('#############')

from imblearn.over_sampling import SMOTE
sos = SMOTE(random_state=42)
X_smote_res, y_smote_res = sos.fit_resample(X, y)
print('SMOTE oversampled dataset shape %s' % Counter(y_smote_res))

X_train, X_test, y_train, y_test = train_test_split(X_smote_res, y_smote_res, test_size=0.5, random_state=1)

#Derive two folds for cross validation
X_train1 = X_train; y_train1 = y_train
X_test1 = X_test; y_test1 = y_test

X_train2 = X_test; y_train2 = y_test
X_test2 = X_train; y_test2 = y_train

print('Number of samples in fold1:{}'.format(len(X_train1)))
print('Number of samples in fold2:{}'.format(len(X_train2)))

# Model multi layer perceptron - neural network
from sklearn.neural_network import MLPClassifier
m3 = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000)

# fold1
clf3 = m3.fit(X_train1,y_train1)
y_pred1 = clf3.predict(X_test1)

# fold2
clf3 = m3.fit(X_train2,y_train2)
y_pred2 = clf3.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = y_raw #le.inverse_transform(y_raw)
y_testr = [*y_test1, *y_test2] #le.inverse_transform([*y_test1, *y_test2])

# Metrics
print("Number of mislabeled points out of a total %d points : %d" % (len(y_testr), len( [i for i, j in zip(y_testr, y_predr) if i != j] )))
        
# Accuracy
acc = accuracy_score(y_testr, y_predr)
print('Accuracy of the {:s} model with SMOTE oversampling: {:f}'.format('Neural network',acc))

# Confusion matrix
cmat=confusion_matrix(y_testr, y_predr)
print('Confusion Matrix of Neural network with SMOTE oversampling:')
print(cmat)

print('#############')
print('ADASYN oversampling')
print('#############')

from imblearn.over_sampling import ADASYN
aos = ADASYN(sampling_strategy='minority', random_state=42)
X_ADASYN_res, y_ADASYN_res = aos.fit_resample(X, y)
print('ADASYN oversampled dataset shape %s' % Counter(y_ADASYN_res))

X_train, X_test, y_train, y_test = train_test_split(X_ADASYN_res, y_ADASYN_res, test_size=0.5, random_state=1)

#Derive two folds for cross validation
X_train1 = X_train; y_train1 = y_train
X_test1 = X_test; y_test1 = y_test

X_train2 = X_test; y_train2 = y_test
X_test2 = X_train; y_test2 = y_train

print('Number of samples in fold1:{}'.format(len(X_train1)))
print('Number of samples in fold2:{}'.format(len(X_train2)))

# Model multi layer perceptron - neural network
from sklearn.neural_network import MLPClassifier
m4 = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000)

# fold1
clf4 = m4.fit(X_train1,y_train1)
y_pred1 = clf4.predict(X_test1)

# fold2
clf4 = m4.fit(X_train2,y_train2)
y_pred2 = clf4.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = y_raw #le.inverse_transform(y_raw)
y_testr = [*y_test1, *y_test2] #le.inverse_transform([*y_test1, *y_test2])

# Metrics
print("Number of mislabeled points out of a total %d points : %d" % (len(y_testr), len( [i for i, j in zip(y_testr, y_predr) if i != j] )))
        
# Accuracy
acc = accuracy_score(y_testr, y_predr)
print('Accuracy of the {:s} model with ADASYN oversampling: {:f}'.format('Neural network',acc))

# Confusion matrix
cmat=confusion_matrix(y_testr, y_predr)
print('Confusion Matrix of Neural network with ADASYN oversampling:')
print(cmat)

print('*************')
print('Part-3')
print('*************')
from collections import Counter
import imblearn

print('#############')
print('Random undersampling')
print('#############')
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)
print('Random undersampled dataset shape %s' % Counter(y_rus))

X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, test_size=0.5, random_state=1)

#Derive two folds for cross validation
X_train1 = X_train; y_train1 = y_train
X_test1 = X_test; y_test1 = y_test

X_train2 = X_test; y_train2 = y_test
X_test2 = X_train; y_test2 = y_train

print('Number of samples in fold1:{}'.format(len(X_train1)))
print('Number of samples in fold2:{}'.format(len(X_train2)))

# Model multi layer perceptron - neural network
from sklearn.neural_network import MLPClassifier
m5 = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000)

# fold1
clf5 = m5.fit(X_train1,y_train1)
y_pred1 = clf5.predict(X_test1)

# fold2
clf5 = m5.fit(X_train2,y_train2)
y_pred2 = clf5.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = y_raw #le.inverse_transform(y_raw)
y_testr = [*y_test1, *y_test2] #le.inverse_transform([*y_test1, *y_test2])

# Metrics
print("Number of mislabeled points out of a total %d points : %d" % (len(y_testr), len( [i for i, j in zip(y_testr, y_predr) if i != j] )))
        
# Accuracy
acc = accuracy_score(y_testr, y_predr)
print('Accuracy of the {:s} model with Random undersampling: {:f}'.format('Neural network',acc))

# Confusion matrix
cmat=confusion_matrix(y_testr, y_predr)
print('Confusion Matrix of Neural network with Random undersampling:')
print(cmat)

print('#############')
print('Cluster undersampling')
print('#############')
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=42)
X_cc, y_cc = cc.fit_resample(X, y)
print('Cluster undersampled dataset shape %s' % Counter(y_cc))

X_train, X_test, y_train, y_test = train_test_split(X_cc, y_cc, test_size=0.5, random_state=1)

#Derive two folds for cross validation
X_train1 = X_train; y_train1 = y_train
X_test1 = X_test; y_test1 = y_test

X_train2 = X_test; y_train2 = y_test
X_test2 = X_train; y_test2 = y_train

print('Number of samples in fold1:{}'.format(len(X_train1)))
print('Number of samples in fold2:{}'.format(len(X_train2)))

# Model multi layer perceptron - neural network
from sklearn.neural_network import MLPClassifier
m6 = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000)

# fold1
clf6 = m6.fit(X_train1,y_train1)
y_pred1 = clf6.predict(X_test1)

# fold2
clf6 = m6.fit(X_train2,y_train2)
y_pred2 = clf6.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = y_raw #le.inverse_transform(y_raw)
y_testr = [*y_test1, *y_test2] #le.inverse_transform([*y_test1, *y_test2])

# Metrics
print("Number of mislabeled points out of a total %d points : %d" % (len(y_testr), len( [i for i, j in zip(y_testr, y_predr) if i != j] )))
        
# Accuracy
acc = accuracy_score(y_testr, y_predr)
print('Accuracy of the {:s} model with Cluster undersampling: {:f}'.format('Neural network',acc))

# Confusion matrix
cmat=confusion_matrix(y_testr, y_predr)
print('Confusion Matrix of Neural network with Cluster undersampling:')
print(cmat)

print('#############')
print('Tomek Links undersampling')
print('#############')
from imblearn.under_sampling import TomekLinks

tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X, y)
print('Tomek Links undersampled dataset shape %s' % Counter(y_tl))

X_train, X_test, y_train, y_test = train_test_split(X_tl, y_tl, test_size=0.5, random_state=1)

#Derive two folds for cross validation
X_train1 = X_train; y_train1 = y_train
X_test1 = X_test; y_test1 = y_test

X_train2 = X_test; y_train2 = y_test
X_test2 = X_train; y_test2 = y_train

print('Number of samples in fold1:{}'.format(len(X_train1)))
print('Number of samples in fold2:{}'.format(len(X_train2)))

# Model multi layer perceptron - neural network
from sklearn.neural_network import MLPClassifier
m7 = MLPClassifier(solver='adam', alpha=1e-5, max_iter=1000)

# fold1
clf7 = m7.fit(X_train1,y_train1)
y_pred1 = clf7.predict(X_test1)

# fold2
clf7 = m7.fit(X_train2,y_train2)
y_pred2 = clf7.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = y_raw #le.inverse_transform(y_raw)
y_testr = [*y_test1, *y_test2] #le.inverse_transform([*y_test1, *y_test2])

# Metrics
print("Number of mislabeled points out of a total %d points : %d" % (len(y_testr), len( [i for i, j in zip(y_testr, y_predr) if i != j] )))
        
# Accuracy
acc = accuracy_score(y_testr, y_predr)
print('Accuracy of the {:s} model with Tomek Links undersampling: {:f}'.format('Neural network',acc))

# Confusion matrix
cmat=confusion_matrix(y_testr, y_predr)
print('Confusion Matrix of Neural network with Tomek Links undersampling:')
print(cmat)