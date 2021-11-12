#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:42:41 2021

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

import imblearn

import matplotlib



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score


path =r'file:///Users/ritikanair/Dropbox/EECS%20658%20Intro%20to%20ML/iris.csv'
dataset = pd.read_csv(path) #to read the data from iris.csv into the variable 'dataset'

X = dataset
y = dataset.pop('class')

#print("X = \n",X)
#print("y = \n",y)

print("\nPart 1: k-Means Clustering\n")
from sklearn.cluster import KMeans
rec = []
SqEr=[]

for k  in range(1,21):# i values are 1,2, ..., 20
   kmeans = KMeans(n_clusters=k).fit(X)
   #lab=kmeans.labels_
   #means=kmeans.cluster_centers_
  # clcenter = kmeans.predict(X)
   #sqr_err = 0
    
   # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
  # for i in range(len(X)):
   #   center = means[clcenter[i]]
    #  sqr_err = sqr_err + (X[i, 0] - center[0]) ** 2 + (X[i, 1] - center[1]) ** 2
      
   #SqEr.append(sqr_err)
   
   rec_err = kmeans.inertia_ #reconstruction error
   rec.append(rec_err)
  # print("labels", lab)
  
  #print("\n centers", means)

#print(rec)


from matplotlib import pyplot as plt

def plot_graph(arr, name):
    plt.plot(range(1, 21), arr, marker='o')
    plt.title(name + ' vs. k')
    plt.xlabel('k')
    plt.xticks(np.arange(1, 21, 1))
    plt.ylabel(name)
    plt.show()

# Example where: recons_err_arr is an array containing the reconstruction error for various values of k

plot_graph(rec, 'Reconstruction Error')

print("\n Based on graph, manually observed elbow is 3 \n")

#Calculating elbow algorithmically
d=0 

for i in range (1,21):
    d=abs(abs(rec[i]-rec[i-1]) - abs(rec[i+1]-rec[i]))
    if (d < 50): #threshold value is 50
        break
elbow_k = i
print("\n elbow from algorithm = ", elbow_k)
          
print("\n For the above elbow value, confusion matrix and accuracy score is as follows:\n")


iris=[]

meansk = KMeans(n_clusters = elbow_k, random_state=0).fit(X)
predX = meansk.predict(X)
#print("\n Predict for k = 3:", predX)
#setosa = 0; virginica = 0; versicolor = 0

for k in range(0,3):
    setosa = 0; virginica = 0; versicolor = 0
    for i in range(0,150):
        if predX[i]==k:
            if y[i]== "Iris-setosa":
                setosa = setosa + 1
            elif y[i]== "Iris-versicolor":
                versicolor = versicolor + 1
            else:
                virginica = virginica + 1
    m = max(setosa, versicolor, virginica)
    if m == setosa:
        iris.append("Iris-setosa")
    if m == versicolor:
        iris.append("Iris-versicolor")
    if m == virginica:
        iris.append("Iris-virginica")
    #print("\n label = ",k)
    #print("\n m",m)
    #print(iris)

predXstr = []

for i in range(0,150):
    #print("\n i=",i)
    #print("\n predX = ",predX[i])
    #print("\n iris=",iris[i])
    for k in range(0,3):
        if predX[i]== k:
            predXstr.append(iris[k])
   
#print("\n predXstring =\n ", predXstr)
model_accuracy = accuracy_score(y,predXstr)
print("\n accuracy score =",model_accuracy)
model_confusionmatrix = confusion_matrix(y, predXstr)
print("\n confusion matrix:\n ", model_confusionmatrix)

print("\n For elbow value 3, the accuracy score and confusion matrix  are as follows:\n")

iris=[]

meansk = KMeans(n_clusters = 3, random_state=0).fit(X)
predX = meansk.predict(X)
#print("\n Predict for k = 3:", predX)
#setosa = 0; virginica = 0; versicolor = 0

for k in range(0,3):
    setosa = 0; virginica = 0; versicolor = 0
    for i in range(0,150):
        if predX[i]==k:
            if y[i]== "Iris-setosa":
                setosa = setosa + 1
            elif y[i]== "Iris-versicolor":
                versicolor = versicolor + 1
            else:
                virginica = virginica + 1
    m = max(setosa, versicolor, virginica)
    if m == setosa:
        iris.append("Iris-setosa")
    if m == versicolor:
        iris.append("Iris-versicolor")
    if m == virginica:
        iris.append("Iris-virginica")
    #print("\n label = ",k)
    #print("\n m",m)
    #print(iris)

predXstr = []

for i in range(0,150):
    #print("\n i=",i)
    #print("\n predX = ",predX[i])
    #print("\n iris=",iris[i])
    for k in range(0,3):
        if predX[i]== k:
            predXstr.append(iris[k])
   
#print("\n predXstring =\n ", predXstr)
model_accuracy = accuracy_score(y,predXstr)
print("\n accuracy score =",model_accuracy)
model_confusionmatrix = confusion_matrix(y, predXstr)
print("\n confusion matrix:\n ", model_confusionmatrix)


################################################################

print("\n Part 2: GMM \n")
from sklearn.mixture import GaussianMixture
aicgm = []
bicgm=[]

for k  in range(1,21):# i values are 1,2, ..., 20
   gm = GaussianMixture(n_components=k,covariance_type='diag', random_state=0).fit(X)
   gmean=gm.means_
   
   #aic_k = aic(X)
   aicgm.append(gm.aic(X))
   bicgm.append(gm.bic(X))
   #print("\ngmean =",gmean)
  
  #print("\n centers", means)

#print(rec)

#from matplotlib import pyplot as plt

print("\n Using AIC: \n")

def plot_graph(arr, name):
    plt.plot(range(1, 21), arr, marker='o')
    plt.title(name + ' vs. k')
    plt.xlabel('k')
    plt.xticks(np.arange(1, 21, 1))
    plt.ylabel(name)
    plt.show()
plot_graph(aicgm, 'AIC')

print("\n Based on graph, manually observed elbow is 3 \n")

#Algorithm 
d=0

for i in range (1,21):
    d=abs(abs(aicgm[i]-aicgm[i-1]) - abs(aicgm[i+1]-aicgm[i]))
    if (d < 50):
        break
aic_elbow_k = i
print("\n elbow from algorithm = ", aic_elbow_k)

print("\n For the above elbow value, confusion matrix and accuracy score is as follows:\n")

iris_gm=[]
gm_3 = GaussianMixture(n_components=aic_elbow_k,covariance_type='diag', random_state=0).fit(X)
predX = gm_3.predict(X)
#print("\n Predict for k = 3:", predX)
#setosa = 0; virginica = 0; versicolor = 0

for k in range(0,3):
    setosa = 0; virginica = 0; versicolor = 0
    for i in range(0,150):
        if predX[i]==k:
            if y[i]== "Iris-setosa":
                setosa = setosa + 1
            elif y[i]== "Iris-versicolor":
                versicolor = versicolor + 1
            else:
                virginica = virginica + 1
    m = max(setosa, versicolor, virginica)
    if m == setosa:
        iris_gm.append("Iris-setosa")
    if m == versicolor:
        iris_gm.append("Iris-versicolor")
    if m == virginica:
        iris_gm.append("Iris-virginica")
    #print("\n label = ",k)
    #print("\n m",m)
    #print(iris_gm)

predXstr = []

for i in range(0,150):
    #print("\n i=",i)
    #print("\n predX = ",predX[i])
    #print("\n iris=",iris_bic[i])
    for k in range(0,3):
        if predX[i]== k:
            predXstr.append(iris_gm[k])
   
#print("\n predXstring =\n ", predXstr)
model_accuracy = accuracy_score(y,predXstr)
print("\n accuracy score =",model_accuracy)
model_confusionmatrix = confusion_matrix(y, predXstr)
print("\n confusion matrix:\n ", model_confusionmatrix)


         

print("\n Using BIC: \n")


def plot_graph(arr, name):
    plt.plot(range(1, 21), arr, marker='o')
    plt.title(name + ' vs. k')
    plt.xlabel('k')
    plt.xticks(np.arange(1, 21, 1))
    plt.ylabel(name)
    plt.show()
plot_graph(bicgm, 'BIC')

print("\n Based on graph, manually observed elbow is 3 \n")

#Algorithm 
d=0

for i in range (1,21):
    d=abs(abs(bicgm[i]-bicgm[i-1]) - abs(bicgm[i+1]-bicgm[i]))
    if (d < 50):
        break
bic_elbow_k = i
print("\n elbow from algorithm = ", bic_elbow_k)

print("\n For the above elbow value, confusion matrix and accuracy score is as follows:\n")
iris_gm=[]
gm_3 = GaussianMixture(n_components=bic_elbow_k,covariance_type='diag', random_state=0).fit(X)
predX = gm_3.predict(X)
#print("\n Predict for k = 3:", predX)
#setosa = 0; virginica = 0; versicolor = 0

for k in range(0,3):
    setosa = 0; virginica = 0; versicolor = 0
    for i in range(0,150):
        if predX[i]==k:
            if y[i]== "Iris-setosa":
                setosa = setosa + 1
            elif y[i]== "Iris-versicolor":
                versicolor = versicolor + 1
            else:
                virginica = virginica + 1
    m = max(setosa, versicolor, virginica)
    if m == setosa:
        iris_gm.append("Iris-setosa")
    if m == versicolor:
        iris_gm.append("Iris-versicolor")
    if m == virginica:
        iris_gm.append("Iris-virginica")
    #print("\n label = ",k)
    #print("\n m",m)
    #print(iris_gm)

predXstr = []

for i in range(0,150):
    #print("\n i=",i)
    #print("\n predX = ",predX[i])
    #print("\n iris=",iris_bic[i])
    for k in range(0,3):
        if predX[i]== k:
            predXstr.append(iris_gm[k])
   
#print("\n predXstring =\n ", predXstr)
model_accuracy = accuracy_score(y,predXstr)
print("\n accuracy score =",model_accuracy)
model_confusionmatrix = confusion_matrix(y, predXstr)
print("\n confusion matrix:\n ", model_confusionmatrix)



print("\n For elbow value 3, the accuracy score and confusion matrix  are as follows:\n")


iris_gm=[]

gm_3 = GaussianMixture(n_components=3,covariance_type='diag', random_state=0).fit(X)
predX = gm_3.predict(X)
#print("\n Predict for k = 3:", predX)
#setosa = 0; virginica = 0; versicolor = 0

for k in range(0,3):
    setosa = 0; virginica = 0; versicolor = 0
    for i in range(0,150):
        if predX[i]==k:
            if y[i]== "Iris-setosa":
                setosa = setosa + 1
            elif y[i]== "Iris-versicolor":
                versicolor = versicolor + 1
            else:
                virginica = virginica + 1
    m = max(setosa, versicolor, virginica)
    if m == setosa:
        iris_gm.append("Iris-setosa")
    if m == versicolor:
        iris_gm.append("Iris-versicolor")
    if m == virginica:
        iris_gm.append("Iris-virginica")
    #print("\n label = ",k)
    #print("\n m",m)
    #print(iris_gm)

predXstr = []

for i in range(0,150):
    #print("\n i=",i)
    #print("\n predX = ",predX[i])
    #print("\n iris=",iris_bic[i])
    for k in range(0,3):
        if predX[i]== k:
            predXstr.append(iris_gm[k])
   
#print("\n predXstring =\n ", predXstr)
model_accuracy = accuracy_score(y,predXstr)
print("\n accuracy score =",model_accuracy)
model_confusionmatrix = confusion_matrix(y, predXstr)
print("\n confusion matrix:\n ", model_confusionmatrix)











