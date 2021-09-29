# Deep-belief-network

# Import relevant libraries
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

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from dbn import SupervisedDBNClassification
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Loading dataset
digits = load_digits()
X, Y = digits.data, digits.target

# Data scaling
X = (X / 16).astype(np.float32)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)


# Fit
classifier.fit(X_train, Y_train)

# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# Test
Y_pred = classifier.predict(X_test)

# Metrics
print("Number of mislabeled points out of a total %d points : %d" % (len(Y_test), (Y_test != Y_pred).sum()))
        
# Accuracy
acc = accuracy_score(Y_test, Y_pred)
print('Accuracy of the {:s} model: {:f}'.format('Deep Belief Network Classifier',acc))

# Confusion matrix
cm=confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['0','1','2','3','4','5','6','7','8','9'])
_=disp.plot() 
disp.ax_.set(title='Deep Belief Network Classifier')
print()