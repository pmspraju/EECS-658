# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 18:44:34 2021

@author: pmspr
"""

# Import relevant libraries
import os
import sys
import random
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

def computeGs(sar_value, gamma):
    
    # (-1)*0.9^0 + (-1)*0.9 + (-1)*0.9^2..
    return sum([val['r']*(gamma ** i) for i, val in enumerate(sar_value)])

def getaction(i,j,vs,actions):
    
    up    = vs[i-1][j] if ((i-1) >= 0) else -1000 #vs[i][j]
    down  = vs[i+1][j] if ((i+1) <= 4) else -1000 #vs[i][j]
    right = vs[i][j+1] if ((j+1) <= 4) else -1000 #vs[i][j]
    left  = vs[i][j-1] if ((j-1) >= 0) else -1000 #vs[i][j]
    
    if(up!=0 and down!=0 and right!=0 and left!=0):
        return actions[np.argmax([up,down,right,left])]
    else:
        return random.choice(actions)
    
print('*************')
print('Part-1-Monte Carlo First-visit')
print('*************')

actions = ['u', 'd', 'r', 'l'] #up down right left
reward = -1
gamma = 0.9 # discount factor

states = np.array([[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19], [20,21,22,23,0]])
rewrds = np.array([[0,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,0]], dtype=float)

ns = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]], dtype=float)
ss = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]], dtype=float)
vs = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]], dtype=float)

print('************')
print('Epoch-0')
print('************')
print('----N(s)-----')
print(ns)
print('----S(s)-----')
print(ss)
print('----V(s)-----')
print(vs)

itrs = 0
while (itrs <= 100):
    
    rs = random.choice(states.ravel()[1:-1])
    
    ind = np.where(states == rs)
    
    state_reward_action_pair = []; i=ind[0][0]; j=ind[1][0]
    while not ((i==0 and j==0) or (i==4 and j==4)):
        
        ract = random.choice(actions)
        #ract = getaction(i,j,vs,actions)

        state_reward_action_pair.append({'s':states[i][j], 'r':rewrds[i][j], 'act':ract})
        
        if (ract == "u"):
            if ((i-1) >= 0):
                i=i-1
        if (ract == "d"):
            if ((i+1) <= 4):
                i=i+1         
        if (ract == "r"):
            if ((j+1) <= 4):
                j=j+1
        if (ract == "l"):
            if ((j-1) >= 0):
                j=j-1
    
    # Create a tuple like (state, reward, action, G(s))
    gslist = [dict(item, **{'k':i+1,'gamma':gamma ,'gs':computeGs(state_reward_action_pair[i:], gamma)} ) for i, item in enumerate(state_reward_action_pair)]
    
    mcDataframe = pd.DataFrame(gslist)
    mcDataframe = mcDataframe[['k','s','r','gamma','gs']]
    
    # Get unique states 
    uniq_states = list(set([tup['s'] for tup in state_reward_action_pair]))
    
    # Compute N(s) and S(s)
    for ust in uniq_states:
        uin = np.where(states == ust)
        ui=uin[0][0]; uj=uin[1][0]
        
        #N(s) - For first-visit, every state is given 1 for a given epoch
        ns[ui][uj] = ns[ui][uj] + 1
        
        #S(s)
        for ugs in gslist:
            if(ugs['s'] == ust):
                ss[ui][uj] = ss[ui][uj] + ugs['gs']
                break # For First-visit, we break after visiting the state for the first time
    
    # v(s) = s(s)/n(s)
    vs = np.divide(ss, ns, out=np.zeros_like(ss), where=ns!=0)
    
    itrs = itrs+1
    
    if (itrs in [1,10,100]):
        print('************')
        print('Epoch-',itrs)
        print('************')
        print('----N(s)-----')
        print(ns)
        print('----S(s)-----')
        print(ss)
        print('----V(s)-----')
        print(vs) 
        print('--------------------')
        print('k, s, r, γ, and G(s)')
        print('--------------------')
        print(mcDataframe.to_string(index=False))
    
print('*************')
print('Part-2-Monte Carlo Every-visit')
print('*************')

actions = ['u', 'd', 'r', 'l'] #up down right left
reward = -1
gamma = 0.9 # discount factor

states = np.array([[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19], [20,21,22,23,0]])
rewrds = np.array([[0,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,0]], dtype=float)

ns = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]], dtype=float)
ss = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]], dtype=float)
vs = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]], dtype=float)

print('************')
print('Epoch-0')
print('************')
print('----N(s)-----')
print(ns)
print('----S(s)-----')
print(ss)
print('----V(s)-----')
print(vs)

itrs = 0
while (itrs <= 100):
    
    rs = random.choice(states.ravel()[1:-1])
    
    ind = np.where(states == rs)
    
    state_reward_action_pair = []; i=ind[0][0]; j=ind[1][0]
    while not ((i==0 and j==0) or (i==4 and j==4)):
        
        ract = random.choice(actions)
        #ract = getaction(i,j,vs,actions)
        
        state_reward_action_pair.append({'s':states[i][j], 'r':rewrds[i][j], 'act':ract})
        
        if (ract == "u"):
            if ((i-1) >= 0):
                i=i-1
        if (ract == "d"):
            if ((i+1) <= 4):
                i=i+1         
        if (ract == "r"):
            if ((j+1) <= 4):
                j=j+1
        if (ract == "l"):
            if ((j-1) >= 0):
                j=j-1
    
    # Create a tuple like (state, reward, action, G(s))
    gslist = [dict(item, **{'k':i+1,'gamma':gamma ,'gs':computeGs(state_reward_action_pair[i:], gamma)} ) for i, item in enumerate(state_reward_action_pair)]
    
    mcDataframe = pd.DataFrame(gslist)
    mcDataframe = mcDataframe[['k','s','r','gamma','gs']]
    
    # Get unique states 
    uniq_states = list(set([tup['s'] for tup in state_reward_action_pair]))
    
    # Compute N(s) and S(s)
    for ust in uniq_states:
        uin = np.where(states == ust)
        ui=uin[0][0]; uj=uin[1][0]
        
        #S(s) N(s) - count for every visit of the state
        for ugs in gslist:
            if(ugs['s'] == ust):
                ss[ui][uj] = ss[ui][uj] + ugs['gs']
                ns[ui][uj] = ns[ui][uj] + 1
    
    # v(s) = s(s)/n(s)
    vs = np.divide(ss, ns, out=np.zeros_like(ss), where=ns!=0)
    
    itrs = itrs+1
    
    if (itrs in [1,10,100]):
        print('************')
        print('Epoch-',itrs)
        print('************')
        print('----N(s)-----')
        print(ns)
        print('----S(s)-----')
        print(ss)
        print('----V(s)-----')
        print(vs) 
        print('--------------------')
        print('k, s, r, γ, and G(s)')
        print('--------------------')
        print(mcDataframe.to_string(index=False))
    
def changeToDf(nparr):
    df_list = []
    for i in range(nparr.shape[0]):
        rdict = {}
        for j in range(nparr.shape[1]):
            rdict[j] = nparr[i][j]
        df_list.append(rdict)
    df = pd.DataFrame(df_list)
    return df

print('*************')
print('Part-3-Q-Learning')
print('*************')

gamma = 0.9 # discount factor

states = np.array([[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19], [20,21,22,23,24]])
rewards =np.array([
        [100, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [100,-1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [100, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, 100],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 100],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 100]
                  ], dtype=np.int32)

print('Q-Learning Rewards Matrix (R)')
print('-----------------------------')
print(changeToDf(rewards).to_string())
print()

print('Q-Learning Value Matrix (Q)')
print('---------------------------')
print('Initial values')
print('--------------')
Qvalue = np.zeros_like(rewards)
print(changeToDf(Qvalue).to_string())
print()

itrs = 0
while (itrs <= 500):
    
    # Choose random state
    rs = random.choice(states.ravel())
    
    ract_reward = 0
    while (ract_reward != 100):
        
        # Choose random valid action
        ind  = np.where(rewards[rs,:] != -1)
        ract  = random.choice(ind[0])
        ract_reward = rewards[rs][ract]

        # Compute Q-value of (state,action) pair
        # Q(s,a) = r(s,a) + gamma*[Max[Q(s',a')]]
        next_state = ract
        next_actions = np.where(rewards[next_state,:] != -1)
        max_allactions = max([Qvalue[next_state][i] for i in next_actions[0]])
        Qvalue[rs][ract] = ract_reward + gamma*max_allactions
        
        rs = next_state
    
    itrs += 1
    
    if(itrs in [1,10,500]):
        print('Iteration:',itrs)
        print('---------------')
        print(changeToDf(Qvalue).to_string())
        print()
        
print('*************')
print('Part-4-SARSA')
print('*************')

gamma = 0.9 # discount factor

states = np.array([[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19], [20,21,22,23,24]])
rewards =np.array([
        [100, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [100,-1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [100, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, 100],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 100],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 100]
                  ], dtype=np.int32)

print('Q-Learning Rewards Matrix (R)')
print('-----------------------------')
print(changeToDf(rewards).to_string())
print()

print('Q-Learning Value Matrix (Q)')
print('---------------------------')
print('Initial values')
print('--------------')
Qvalue = np.zeros_like(rewards)
print(changeToDf(Qvalue).to_string())
print()

itrs = 0
while (itrs <= 100):
    
    # Choose random state
    rs = random.choice(states.ravel())
    
    ract_reward = 0
    while (ract_reward != 100):
        
        # Choose valid action with max Q-value
        # get the valid actions of the state
        ind  = np.where(rewards[rs,:] != -1)
        #create the dict with {action:q-value}
        act_val = {i:Qvalue[rs][i] for i in ind[0]}
        vals = list(act_val.values()) # dictionary values
        kys =list(act_val.keys()) # dictionary keys
        #get the action with max q-value
        ract = random.choice([kys[j] for j in [i for i,v in enumerate(vals) if(v==max(vals))]])
        ract_reward = rewards[rs][ract] # reward of the (state,action)

        # Compute Q-value of (state,action) pair
        # Q(s,a) = r(s,a) + gamma*[Max[Q(s',a')]]
        next_state = ract
        next_actions = np.where(rewards[next_state,:] != -1)
        max_allactions = max([Qvalue[next_state][i] for i in next_actions[0]])
        Qvalue[rs][ract] = ract_reward + gamma*max_allactions
        
        rs = next_state
    
    itrs += 1
    
    if(itrs in [1,10,100]):
        print('Iteration:',itrs)
        print('---------------')
        print(changeToDf(Qvalue).to_string())
        print()