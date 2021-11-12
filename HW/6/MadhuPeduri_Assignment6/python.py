# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 23:13:56 2021

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
pd.options.mode.chained_assignment = None  # default='warn'
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print('pandas: {}'.format(pd.__version__))

import sklearn
print('sklearn: {}'.format(sklearn.__version__))

from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay, recall_score

print("Hello World!")

# Read the file
from sklearn import datasets

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 6\EECS 658\Data'
filename = "iris.csv"
data = pd.read_csv(os.path.join(path,filename))

y = data.pop('class')
X = data

def re(dfp):
    _ = dfp.pop('clusters')
    m = dfp.describe().loc['mean']
    cols = dfp.columns
    for c in cols:
        dfp.loc[:,c] = dfp.loc[:,c].apply( lambda a: pow( (a-m[c]),2 ) )
    return(dfp.values.sum())
    
def reconErr(df):
    ire = [re(df[df['clusters'] == i]) for i in df['clusters'].unique()]
    return(sum(ire))

from sklearn.cluster import KMeans
def kmeansModel(dat,n_cls):
    model = KMeans(n_clusters=n_cls, random_state=0)
    model.fit(dat)
    clusters = model.predict(dat)
    dat['clusters'] = clusters
    recon_error = reconErr(dat)
    return(recon_error)

print('*************')
print('Part-1-k-Means Clustering')
print('*************')

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# Compute reconstruction error for k= 1 to 20
recon_errlist = [kmeansModel(X,i) for i in range(1,21)]

#plot the elobow curve
_ = plt.plot(range(1,21), recon_errlist, marker='o', color='g')
_=plt.xlabel('K'); _=plt.ylabel('Reconstruction error'); _=plt.title('Elbow curve')
_ =plt.show()

# Calculate the K algorithmically
dif = abs(np.diff(recon_errlist))

elbow_k=0
for i,val in enumerate(dif):
    if (i < (len(dif) - 1)):
        if( (val - dif[i+1]) > np.subtract(*np.percentile(recon_errlist, [50, 25])) ):
            elbow_k=elbow_k+1

print('*************')
print('Appropriate K from elbow curve:', elbow_k)
print('*************')

#k-means clustering using elbow_k clusters
data = pd.read_csv(os.path.join(path,filename))
y = data.pop('class')
X = data

ek_model = KMeans(n_clusters=elbow_k, random_state=0)
ek_model.fit(X)
clusters = ek_model.predict(X)
k_labels = clusters # ek_model.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels, dtype=np.dtype('U25'))

for k in np.unique(k_labels):
    
    # ...find and assign the best-matching truth label
    cpart = np.array(y)[(np.where(k_labels==k)[0])]
    match_nums = [ len(cpart[cpart==t]) for t in np.unique(y) ]
    k_labels_matched[k_labels==k] = np.unique(y)[np.argmax(match_nums)]
      
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y, k_labels_matched)
print('Confusion matrix for k-means clustering using elbow_k clusters:')
print(cm)
print()
print('Accuracy score for k-means with elbow_k clusters:', accuracy_score(y, k_labels_matched))

#k-means clustering using k=3 clusters
data = pd.read_csv(os.path.join(path,filename))
y = data.pop('class')
X = data

k_3 = KMeans(n_clusters=3, random_state=0)
k_3.fit(X)
clusters = k_3.predict(X)

k_labels = clusters # ek_model.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels, dtype=np.dtype('U25'))

# For each cluster label...
for k in np.unique(k_labels):
    
    # ...find and assign the best-matching truth label
    cpart = np.array(y)[(np.where(k_labels==k)[0])]
    match_nums = [ len(cpart[cpart==t]) for t in np.unique(y) ]
    k_labels_matched[k_labels==k] = np.unique(y)[np.argmax(match_nums)]
    
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y, k_labels_matched)
print('Confusion matrix for k-means clustering using k=3 clusters:')
print(cm)
print()
print('Accuracy score for k-means with k=3 clusters:', accuracy_score(y, k_labels_matched))

from sklearn import mixture
def GaussianModel(dat,n_cls):
    gmm = mixture.GaussianMixture(n_components=n_cls, covariance_type="diag")
    gmm.fit(dat)
    
    return(gmm.aic(dat), gmm.bic(dat))

print('*************')
print('Part-2-GMM Clustering')
print('*************')

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

data = pd.read_csv(os.path.join(path,filename))
y = data.pop('class')
X = data

# Compute aic and bic values for k= 1 to 20
aic_bic_list = [GaussianModel(X,i) for i in range(1,21)]
aic_list = [i[0] for i in aic_bic_list]
aic_elbow_k = 3

#plot the elobow curve
_ =plt.plot(range(1,21), aic_list, marker='o', color='g')
_ =plt.xlabel('K'); _=plt.ylabel('AIC'); _=plt.title('AIC vs k')
_ =plt.show()

bic_list = [i[1] for i in aic_bic_list]

#plot the elobow curve
_b = plt.plot(range(1,21), bic_list, marker='o', color='g')
_b=plt.xlabel('K'); _=plt.ylabel('BIC'); _=plt.title('BIC vs k')
_ =plt.show()

# Calculate the K algorithmically
dif = abs(np.diff(bic_list))
bic_elbow_k=0
for i,val in enumerate(dif):
    if( val > np.subtract(*np.percentile(bic_list, [50, 25])) ):
        bic_elbow_k=bic_elbow_k+1

print('*************')
print('Appropriate K from elbow curve:', bic_elbow_k)
print('*************')

#GMM clustering using k=aic_elbow_k clusters
data = pd.read_csv(os.path.join(path,filename))
y = data.pop('class')
X = data

gmm_aic = mixture.GaussianMixture(n_components=aic_elbow_k, covariance_type="diag")
gmm_aic.fit(X)
clusters = gmm_aic.predict(X)

k_labels = clusters # ek_model.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels, dtype=np.dtype('U25'))

# For each cluster label...
for k in np.unique(k_labels):
    
    # ...find and assign the best-matching truth label
    cpart = np.array(y)[(np.where(k_labels==k)[0])]
    match_nums = [ len(cpart[cpart==t]) for t in np.unique(y) ]
    k_labels_matched[k_labels==k] = np.unique(y)[np.argmax(match_nums)]
    
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y, k_labels_matched)
print('Confusion matrix for gmm clustering using k=aic_elbow_k clusters:')
print(cm)
print()
print('Accuracy score for gmm with k=aic_elbow_k clusters:', accuracy_score(y, k_labels_matched))

#GMM clustering using k=bic_elbow_k clusters
data = pd.read_csv(os.path.join(path,filename))
y = data.pop('class')
X = data

gmm_aic = mixture.GaussianMixture(n_components=bic_elbow_k, covariance_type="diag")
gmm_aic.fit(X)
clusters = gmm_aic.predict(X)

k_labels = clusters # ek_model.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels, dtype=np.dtype('U25'))

# For each cluster label...
for k in np.unique(k_labels):
    
    # ...find and assign the best-matching truth label
    cpart = np.array(y)[(np.where(k_labels==k)[0])]
    match_nums = [ len(cpart[cpart==t]) for t in np.unique(y) ]
    k_labels_matched[k_labels==k] = np.unique(y)[np.argmax(match_nums)]
    
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y, k_labels_matched)
print('Confusion matrix for gmm clustering using k=bic_elbow_k clusters:')
print(cm)
print()
print('Accuracy score for gmm with k=bic_elbow_k clusters:', accuracy_score(y, k_labels_matched))

#GMM clustering using k=3 clusters
data = pd.read_csv(os.path.join(path,filename))
y = data.pop('class')
X = data

gmm_aic = mixture.GaussianMixture(n_components=3, covariance_type="diag")
gmm_aic.fit(X)
clusters = gmm_aic.predict(X)

k_labels = clusters # ek_model.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels, dtype=np.dtype('U25'))

# For each cluster label...
for k in np.unique(k_labels):
    
    # ...find and assign the best-matching truth label
    cpart = np.array(y)[(np.where(k_labels==k)[0])]
    match_nums = [ len(cpart[cpart==t]) for t in np.unique(y) ]
    k_labels_matched[k_labels==k] = np.unique(y)[np.argmax(match_nums)]
    
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y, k_labels_matched)
print('Confusion matrix for gmm clustering using k=3 clusters:')
print(cm)
print()
print('Accuracy score for gmm with k=3 clusters:', accuracy_score(y, k_labels_matched))