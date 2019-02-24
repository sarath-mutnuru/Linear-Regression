#!/bin/python3

import math
import os
import random
import re
import sys
import numpy as np



if __name__ == '__main__':
    
    #timetest=float(input())
    train=open('train.txt','r')
    X=[]
    y=[]
    for line in train:
        i=[float(k) for k in line.split(',')]
        X.append([i[0],i[0]**2])
        y.append(i[-1])
    train.close()
    n=len(X)
    X=np.asarray(X).reshape((n,2))
    o=np.ones((n,1))
    X=np.concatenate((o,X),axis=1)
    y=np.asarray(y).reshape((n,1))
    
    #%%
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 21)
    X_train_l=X_train[:,0:-1]
    beta_l=np.linalg.inv(np.matmul(X_train_l.transpose(),X_train_l))
    beta_l=np.matmul(np.matmul(beta_l,X_train_l.transpose()),y_train)
    beta_q=np.linalg.inv(np.matmul(X_train.transpose(),X_train))
    beta_q=np.matmul(np.matmul(beta_q,X_train.transpose()),y_train)
    # testing on validation set
    y_val_pred_l=np.matmul(X_val[:,0:-1],beta_l)
    MSE_val_l=np.mean((y_val-y_val_pred_l)**2)
    print('MSE of validation using linear is',MSE_val_l)
    y_val_pred_q=np.matmul(X_val,beta_q)
    MSE_val_q=np.mean((y_val-y_val_pred_q)**2)
    print('MSE of validation using quad is',MSE_val_q)
    
    #%%
    #plotting data
    import matplotlib.pyplot as plt
    plt.scatter(X[:,1],y,c='b',marker='x',label='actual')
    X_l=X[:,0:-1]
    y_pred_l=np.matmul(X_l,beta_l)
    y_pred_q=np.matmul(X,beta_q)
    plt.scatter(X[:,1],y_pred_q,c='g', marker=(5,2), label='predicted using quad')
    plt.scatter(X[:,1],y_pred_l,c='r', marker='+', label='predicted using linear')
    
    plt.legend(loc='upper left')
    plt.show()
    #%%
    
    timeCharged = float(input())
    x_test=[1]+[timeCharged,timeCharged**2]
    x_test=np.asarray(x_test).reshape((len(x_test),1))
    y_pred_l=np.matmul(beta_l.transpose(),x_test[0:-1,:])[0][0]
    y_pred_q=np.matmul(beta_q.transpose(),x_test)[0][0]
    print ('using quad ',y_pred_q)
    print ('using linear ',y_pred_l)
