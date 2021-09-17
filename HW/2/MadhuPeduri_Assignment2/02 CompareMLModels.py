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
           print ("-----------------------------------------------------------------------")
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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['Iris-setosa','Iris-versicolor','Iris-virginica'])
        _=disp.plot()
        disp.ax_.set(title=modelname)
        print()
        
            
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
# Read the file
path = r'C:\Users\pmspr\Documents\HS\MS\Sem 6\EECS 658\Data'
filename = "iris.csv"
data = loadData(path,filename)

X = data
y = data.pop('class')
y_asis = y

# Encode the label
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)

print('Classes of the label:')
print(le.classes_)

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

# Model-1-Linear regression
from sklearn.linear_model import LinearRegression
m1 = LinearRegression()

# fold1
clf1 = m1.fit(X_train1,y_train1)
y_pred1 = clf1.predict(X_test1)

# fold2
clf1 = m1.fit(X_train2,y_train2)
y_pred2 = clf1.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = le.inverse_transform([round(x) for x in [*y_pred1, *y_pred2]])
y_testr = le.inverse_transform([*y_test1, *y_test2])

# Metrics
metrics(y_testr,y_predr,'Linear Regression')

# Model-2-Polynomial regression with degree 2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create a pipeline 
# step1 - Create polynomial features
# step2 - Use linear regression
m2 = Pipeline([('poly', PolynomialFeatures(degree=2)),
                  ('linear', LinearRegression(fit_intercept=False))])

# fold1
clf2 = m2.fit(X_train1,y_train1)
y_pred1 = clf2.predict(X_test1)

# fold2
clf2 = m2.fit(X_train2,y_train2)
y_pred2 = clf2.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = [round(x) for x in [*y_pred1, *y_pred2]]
y_predr = le.inverse_transform([2 if v >= 3 else 0 if v < 0 else v for v in y_predr])
y_testr = le.inverse_transform([*y_test1, *y_test2])

# Metrics
metrics(y_testr,y_predr, 'Polynomial Regression with degree2')

# Model-3-Polynomial regression with degree 3
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create a pipeline 
# step1 - Create polynomial features
# step2 - Use linear regression
m3 = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression(fit_intercept=False))])

# fold1
clf3 = m3.fit(X_train1,y_train1)
y_pred1 = clf3.predict(X_test1)

# fold2
clf3 = m3.fit(X_train2,y_train2)
y_pred2 = clf3.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
y_predr = [round(x) for x in [*y_pred1, *y_pred2]]
y_predr = le.inverse_transform([2 if v >= 3 else 0 if v < 0 else v for v in y_predr])
y_testr = le.inverse_transform([*y_test1, *y_test2])

# Metrics
metrics(y_testr,y_predr, 'Polynomial Regression with degree3')

# Model-4-Naive Bayesian classifier
from sklearn.naive_bayes import GaussianNB
m4 = GaussianNB()
#clf4 = gnb.fit(X_train, y_train)

# fold1
clf4 = m4.fit(X_train1,y_train1)
y_pred1 = clf4.predict(X_test1)

# fold2
clf4 = m4.fit(X_train2,y_train2)
y_pred2 = clf4.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
#y_predr = [round(x) for x in [*y_pred1, *y_pred2]]
y_predr = le.inverse_transform(y_raw)
y_testr = le.inverse_transform([*y_test1, *y_test2])

# Metrics
metrics(y_testr,y_predr, 'Naive Bayes')

# Model-5-K-Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier

m5 = KNeighborsClassifier(n_neighbors=10)

# fold1
clf5 = m5.fit(X_train1,y_train1)
y_pred1 = clf5.predict(X_test1)

# fold2
clf5 = m5.fit(X_train2,y_train2)
y_pred2 = clf5.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
#y_predr = [round(x) for x in [*y_pred1, *y_pred2]]
y_predr = le.inverse_transform(y_raw)
y_testr = le.inverse_transform([*y_test1, *y_test2])

# Metrics
metrics(y_testr,y_predr, 'K Neighbour')

# Model-6-K-LDA (LinearDiscriminantAnalysis) classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

m6 = LinearDiscriminantAnalysis()

# fold1
clf6 = m6.fit(X_train1,y_train1)
y_pred1 = clf6.predict(X_test1)

# fold2
clf6 = m6.fit(X_train2,y_train2)
y_pred2 = clf6.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
#y_predr = [round(x) for x in [*y_pred1, *y_pred2]]
y_predr = le.inverse_transform(y_raw)
y_testr = le.inverse_transform([*y_test1, *y_test2])

# Metrics
metrics(y_testr,y_predr, 'Linear Discriminant Analysis')

# Model-7-K-QDA (QuadraticDiscriminantAnalysis) classifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

m7 = QuadraticDiscriminantAnalysis()

# fold1
clf7 = m7.fit(X_train1,y_train1)
y_pred1 = clf7.predict(X_test1)

# fold2
clf7 = m7.fit(X_train2,y_train2)
y_pred2 = clf7.predict(X_test2)

# Inverse transform
y_raw   = [*y_pred1, *y_pred2]
#y_predr = [round(x) for x in [*y_pred1, *y_pred2]]
y_predr = le.inverse_transform(y_raw)
y_testr = le.inverse_transform([*y_test1, *y_test2])

# Metrics
metrics(y_testr,y_predr, 'Quadratic Discriminant Analysis')