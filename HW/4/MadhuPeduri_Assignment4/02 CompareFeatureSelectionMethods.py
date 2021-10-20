# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 18:00:38 2021

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
           
# Deduce the metrics
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay

def metrics(test,pred,modelname):
    try:
        print("Number of mislabeled points out of a total %d points : %d" % (len(test), (test != pred).sum()))
        
        # Accuracy
        acc = accuracy_score(test, pred)
        print('Accuracy of the {:s} model: {:f}'.format(modelname,acc))
        
        # Confusion matrix
        cm=confusion_matrix(test,pred)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm,
        #                              display_labels=['Iris-setosa','Iris-versicolor','Iris-virginica'])
        #_=disp.plot() 
        #disp.ax_.set(title=modelname)
        print('Confusion Matrix:')
        print(cm)
        print()
            
    except Exception as ex:
           print ("Exception occured in metrics-------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
# Read the file
path = r'C:\Users\pmspr\Documents\HS\MS\Sem 6\EECS 658\Data'
filename = "iris.csv"
data = loadData(path,filename)

y = data.pop('class')
X = data
y_asis = y

# Encode the label
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)

#print('Classes of the label:')
#print(le.classes_)

y = list(le.transform(y))

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

print('##################################')
print('# Part 1')
print('##################################')
# Model-8-Support Vector Machine - Linear SVC
from sklearn.svm import SVC
m1 = SVC(gamma=.1, kernel='linear', probability=True)

# fold1
clf1 = m1.fit(X_train1,y_train1)
y_pred1 = clf1.predict(X_test1)

# fold2
clf1 = m1.fit(X_train2,y_train2)
y_pred2 = clf1.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = le.inverse_transform(y_raw)
y_testr = le.inverse_transform([*y_test1, *y_test2])

# Metrics
metrics(y_testr,y_predr, 'Support Vector Machines - Linear SVC')

part1_features = X.columns

print('##################################')
print('# Part 2 - Principal component analysis')
print('##################################')
import numpy as np
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

X_array = array([list(X.iloc[i,:]) for i in range(150)])

# define a matrix
A = X_array

# calculate the mean of each column
M = mean(A.T, axis=1)

# center columns by subtracting column means
C = A - M
V = np.var(C[:,0])

# calculate covariance matrix of centered matrix
#V = np.cov(C.T, bias=1)
V = np.cov(C.T)
#print('Covariance Matrix:')
#print(V)
#print('-----------')

# eigendecomposition of covariance matrix
values, vectors = eig(V)
print("Eigen Vectors:")
print(vectors)
print('-----------')
print("Eigen Values:")
print(values)
print('Sum:',np.sum(values))
print('-----------')

# project data
P = vectors.T.dot(C.T)
#print("Projected Data")
#print(P.T)
#print('-----------')

# Proportion of variance
L1 = 4.24025608 
SumL = np.sum(values)
pov = L1/SumL
print('Proportion of Variance (PoV)',pov)

# SVM using subset of transformed iris data
iris_subset = P[0]
iris_subset_df = pd.DataFrame(iris_subset, columns=['irisPca'])

X_train, X_test, y_train, y_test = train_test_split(iris_subset_df, y, test_size=0.5, random_state=1)

#Derive two folds for cross validation
X_tr1 = X_train; y_tr1 = y_train
X_tst1 = X_test; y_tst1 = y_test

X_tr2 = X_test; y_tr2 = y_test
X_tst2 = X_train; y_tst2 = y_train

from sklearn.svm import SVC
m2 = SVC(gamma=.1, kernel='linear', probability=True)

# fold1
clf2 = m2.fit(X_tr1,y_tr1)
y_pred1 = clf2.predict(X_tst1)

# fold2
clf2 = m2.fit(X_tr2,y_tr2)
y_pred2 = clf2.predict(X_tst2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = le.inverse_transform(y_raw)
y_testr = le.inverse_transform([*y_tst1, *y_tst2])

# Metrics
metrics(y_testr,y_predr, 'Support Vector Machines - Linear SVC')

part2_features = ['First Principal component']

print('##################################')
print('# Part 3 - Simulated annaling')
print('##################################')
def modelsvm(df,label):
    try:
        X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.5, random_state=1)
        #Derive two folds for cross validation
        X_tr1 = X_train; y_tr1 = y_train
        X_tst1 = X_test; y_tst1 = y_test

        X_tr2 = X_test; y_tr2 = y_test
        X_tst2 = X_train; y_tst2 = y_train

        from sklearn.svm import SVC
        m = SVC(gamma=.1, kernel='linear', probability=True)

        # fold1
        clf = m.fit(X_tr1,y_tr1)
        y_pred1 = clf.predict(X_tst1)

        # fold2
        clf = m.fit(X_tr2,y_tr2)
        y_pred2 = clf.predict(X_tst2)

        # Inverse transform
        y_raw   = [*y_pred1, *y_pred2]
        y_predr = le.inverse_transform(y_raw)
        y_testr = le.inverse_transform([*y_tst1, *y_tst2])
        
        # Metrics
        acc = accuracy_score(y_testr, y_predr)
        cmsv=confusion_matrix(y_testr, y_predr)
        
        return acc, cmsv 
    except Exception as ex:
           print ("Exception occured in svm model-------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
pca_df = pd.DataFrame(P.T, columns=['sepal-length-pca','sepal-width-pca','petal-length-pca','petal-width-pca'])
icol = ['sepal-length','sepal-width','petal-length','petal-width','sepal-length-pca','sepal-width-pca','petal-length-pca','petal-width-pca']
iris_pca = pd.concat([X, pca_df], axis=1) 

# Simulated annealing
import random
iters = 100
accepted_accuracy = 0
accepted_subset = []
current_subset = icol #random.sample(icol,2)
stat = ' '; rr=0; Pr_accept = 0
restart_counter= 0
best_accuracy= 0
best_feature_subset= current_subset
##random.seed(23)

for i in range(iters):
    print('Iteration:',i)
    if(random.choice([0,1]) == 0):#add
        
        # Features excluding the current subset
        nodup_list = list( set(icol).difference(current_subset) )
        
        if (len(nodup_list) > 1):
            plist =random.sample(nodup_list, random.choice([1,2]) )
            current_subset = current_subset + plist
            
        elif (len(nodup_list) == 1):
            plist = nodup_list
            current_subset = current_subset + plist
            
        else: #current subset has all the features, so remove random features
            plist =random.sample( icol, random.choice([1,2]) )
            current_subset = list( set(current_subset).difference(plist) )       
        
    else:#delete
    
        if (len(current_subset) == 0): #if current subset is empty, add randome features
            plist =random.sample( icol, random.choice([1,2]) )
            current_subset = current_subset + plist
        else:
            plist =random.sample( icol, random.choice([1,2]) )
            current_subset = list( set(current_subset).difference(plist) )
    
    # discard the empty set
    if (len(current_subset) == 0):
        print('Empty set - discard')
        print('---------------------------------------------')
        continue
        
    temp_df = iris_pca[current_subset]
    model_acc, cmsv = modelsvm(temp_df,y)
    
    if (accepted_accuracy < model_acc):
        accepted_accuracy = model_acc
        stat = 'Improved'
    else:
        Pr_accept = np.exp( -1 * i * ( (accepted_accuracy - model_acc) / accepted_accuracy ) )
        rr = np.random.random()
        if (rr > Pr_accept):
            current_subset = accepted_subset #reject the new dataset
            stat = 'Reject'
        else:
            accepted_accuracy = model_acc
            stat = 'Accept'
    
    # Restart logic
    if (model_acc > best_accuracy):
        best_accuracy = model_acc
        best_feature_subset = current_subset
        restart_counter= 0
    else:
        restart_counter = restart_counter + 1
        if (restart_counter == 10):
            current_subset = best_feature_subset
            model_acc = best_accuracy
            restart_counter= 0
            stat = 'Restart'
            
    accepted_subset = current_subset
    accepted_accuracy = model_acc
    
    print('Accepted Features:',accepted_subset)
    print('Accuracy:',model_acc)
    print('Pr[accept]:',Pr_accept)
    print('Random uniform:',rr)
    print('Status:',stat)
    print('---------------------------------------------')

part3_features = accepted_subset
print('Confusion Matrix - Simulated annealing:')
print(cmsv)
    
print('##################################')
print('# Part 4 - Genetic algorithm')
print('##################################')
# crossovers
from itertools import combinations
def union(t):
    u = t[0] + t[1]
    return list(set(u))

def intersection(t):
    return list(set(t[0]) & set(t[1]))
    
def crossovers(actual):
    try:
        
        comb = list(combinations(actual,2))
        u = [union(i) for i in comb]
        i = [intersection(i) for i in comb if (len(intersection(i)) > 0)]
            
        return (u + i)
        
    except Exception as ex:
           print ("Exception occured in crossovers-------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
        
def mutations(crv):
    try:
        act = random.choice([0,1,2])
        
        if (act == 0): #add
            
            # Features excluding the current list
            nodup_list = list( set(icol).difference(crv) )
            
            if (len(nodup_list) > 0):
                plist =random.sample(nodup_list, 1)
                crv = crv + plist
            else:#if list is having all the features, delete a random feature
                plist =random.sample( icol, 1 )
                crv = list( set(crv).difference(plist) )
                
        if (act == 1): #delete
            
            if (len(crv) <= 1): #if current subset is empty, add randome features
                plist =random.sample( icol, 1 )
                crv = crv + plist
            else:
                plist =random.sample( crv, 1 )
                crv = list( set(crv).difference(plist) )
                
        if (act == 2): #replace
            plist = random.sample(crv,1)
            crv = list( set(crv).difference(plist) )
            plist =random.sample( icol, 1 )
            crv = crv + plist
                
        return crv
        
    except Exception as ex:
           print ("Exception occured in mutations-------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
        
def evalsvm(eval_set):
    try:
        acc_list = []
        for eset in eval_set:
            temp_df = iris_pca[eset]
            model_acc, cmsv = modelsvm(temp_df,y)
            adict = {}
            adict['Best set'] = eset
            adict['accuracy'] = model_acc
            adict['cm'] = cmsv
            acc_list.append(adict)
            
        return acc_list
            
    except Exception as ex:
           print ("Exception occured in evalsvm-------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
icol = ['sepal-length','sepal-width','petal-length','petal-width','sepal-length-pca','sepal-width-pca','petal-length-pca','petal-width-pca']
init_pop = [['sepal-length-pca', 'sepal-length', 'sepal-width', 'petal-length', 'petal-width'],
            ['sepal-length-pca','sepal-width-pca', 'sepal-width', 'petal-length', 'petal-width'],
            ['sepal-length-pca','sepal-width-pca','petal-length-pca', 'sepal-width', 'petal-length'],
            ['sepal-length-pca','sepal-width-pca','petal-length-pca','petal-width-pca', 'sepal-width'],
            ['sepal-length-pca','sepal-width-pca','petal-length-pca','petal-width-pca', 'sepal-length']]
gen = 50

cv = crossovers(init_pop)
mv = [mutations(x) for x in cv]
eval_set = init_pop + cv + mv

#display(acc_df)

for g in range(gen):
    
    if (g > 0):
        acc_list = evalsvm(eval_set)
        acc_df = pd.DataFrame(acc_list).sort_values('accuracy',ascending=False)
        
        print('*****************')
        print('Generation:', g)
        print('*****************')
        #with pd.option_context('max_colwidth', 1000):
        fi = 0
        for fi in range(5):
            print('....................')
            print('Feature set',acc_df.iloc[fi,0])
            print('Accuracy', acc_df.iloc[fi,1])
            
            if (g == 49 and fi ==0):
                part4_features = acc_df.iloc[fi,0]
                print('Confusion Matrix - Genetic algorithm:')
                print(acc_df.iloc[fi,2])
        
        # Terminate
        if (acc_df['accuracy'][0] >= 1):
            break
        
        # Update the initial population with first five from evalation
        init_pop = []
        for z in range(5):
            init_pop.append(acc_df['Best set'][z])
            
    cv = crossovers(init_pop)
    mv = [mutations(x) for x in cv]
    eval_set = init_pop + cv + mv

print('*****************************')
print('Final Metrics for each part:')
print('*****************************')
print('Part 1 - SVM', part1_features)
print('Part 2 - PCA', part2_features)
print('Part 3 - Simulated annealing', part3_features)
print('Part 4 - Genetic algorithm', part4_features)