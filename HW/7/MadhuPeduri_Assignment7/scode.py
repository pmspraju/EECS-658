# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:34:55 2021

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

def computeprob(k0,i,row,ev,gamma,ps):
    newprob =[]
    for j,val in enumerate(row):
        
        #for termial states, prob=0
        if( (i==0 and j==0) or (i==4 and j==4) ):
            newprob.append(0)
        else:
            up    = k0[i-1][j] if ((i-1) >= 0) else val
            down  = k0[i+1][j] if ((i+1) <= 4) else val
            right = k0[i][j+1] if ((j+1) <= 4) else val
            left  = k0[i][j-1] if ((j-1) >= 0) else val
            prob = ev + gamma*ps*(up+down+right+left)
            
            newprob.append(prob)
            
    return newprob

print('*************')
print('Part-1-Policy Iteration')
print('*************')

actions = ['u', 'd', 'r', 'l'] #up down right left
reward = -1
pr = 0.25 #
ps = 0.25 #state transition probability
gamma = 1 # discount factor

ev = sum([reward * pr for i in actions])

k0 = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]])

itrs = 0
while True:
    
    if(itrs in [0,1,10]):
        print('Policy for Iteration:',itrs)
        print('*************')
        print(k0)
        print('*************')
    
    k1 = np.array([computeprob(k0,i,row,ev,gamma,ps) for i, row in enumerate(k0)])
    converged = True
    for i,j in zip(k0.ravel(),k1.ravel()):
        if(i != j):
            converged = False
            
    if converged:
        print('Optimal Policy Converged in iterations:',itrs)
        print('*************')
        print(k0)
        print('*************')
        break
    else:
        k0 = k1
        itrs = itrs+1
        
def computevalue(k0,i,row,ev,gamma,ps):
    newprob =[]
    for j,val in enumerate(row):
        
        #for termial states, prob=0
        if( (i==0 and j==0) or (i==4 and j==4) ):
            newprob.append(0)
        else:
            up    = ev + gamma*(k0[i-1][j] if ((i-1) >= 0) else val)
            down  = ev + gamma*(k0[i+1][j] if ((i+1) <= 4) else val)
            right = ev + gamma*(k0[i][j+1] if ((j+1) <= 4) else val)
            left  = ev + gamma*(k0[i][j-1] if ((j-1) >= 0) else val)
            prob = max(up,down,right,left)
            
            newprob.append(prob)
            
    return newprob

print('*************')
print('Part-2-Value Iteration')
print('*************')

actions = ['u', 'd', 'r', 'l'] #up down right left
reward = -1
pr = 0.25 #
ps = 0.25 #state transition probability
gamma = 1 # discount factor

ev = sum([reward * pr for i in actions])

k0 = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]])

itrs = 0
while True:
    
    if(itrs in [0,1,2]):
        print('Policy for Iteration:',itrs)
        print('*************')
        print(k0)
        print('*************')
    
    k1 = np.array([computevalue(k0,i,row,ev,gamma,ps) for i, row in enumerate(k0)])
    converged = True
    for i,j in zip(k0.ravel(),k1.ravel()):
        if(i != j):
            converged = False
            
    if converged:
        print('Optimal Policy Converged in iterations:',itrs)
        print('*************')
        print(k0)
        print('*************')
        break
    else:
        k0 = k1
        itrs = itrs+1