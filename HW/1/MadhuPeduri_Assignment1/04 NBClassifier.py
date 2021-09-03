# Import relevant libraries
import os
import sys
print('Python: {}'.format(sys.version))

import scipy
print('scipy: {}'.format(scipy.__version__))

import numpy
print('numpy: {}'.format(numpy.__version__))

import pandas as pd
print('pandas: {}'.format(pd.__version__))

import sklearn
print('sklearn: {}'.format(sklearn.__version__))

#Function to load the data
def loadData(path,filename):
    try:
             files = os.listdir(path)
             for f in files:
                 if f == filename:
                     data = pd.read_csv(os.path.join(path,f))
                     return data
            
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
# Read the file
path = r'C:\Users\pmspr\Documents\HS\MS\Sem 6\EECS 658\Data'
filename = "iris.csv"
data = loadData(path,filename)
display(data)

X = data
y = data.pop('class')

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Train the NB Classifier model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
clf = gnb.fit(X_train, y_train)

# Predict the test data
y_pred = clf.predict(X_test)

# Create a local csv
pd.DataFrame({'actual' : y_test,'pred' : y_pred}).to_csv('test.csv',index=False)

# Deduce the metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,ConfusionMatrixDisplay, classification_report

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

acc = accuracy_score(y_test, y_pred)
print('Accuracy of the NB model: {:f}'.format(acc))

precision = precision_score(y_test, y_pred,average=None)
print('Precision score of the NB model {}'.format(precision))

recall = recall_score(y_test, y_pred,average=None)
print('Recall score of the NB model {}'.format(recall))

f1 = f1_score(y_test, y_pred,average=None)
print('F1 score of the NB model {}'.format(f1))

cm=confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
_=disp.plot() 

print()
print(classification_report(y_test,y_pred))