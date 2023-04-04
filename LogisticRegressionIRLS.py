import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

class LR():
    def __init__(self, maximisation_minorisation = False):
        self.beta = []
        self.maximisation_minorisation = maximisation_minorisation
        
    def sigmoid(self, X, β):
        return scipy.special.expit(X @ β)


    def fit(self, X, Y, iter_max = 100, interactions = [], l2=0.05): 
        
        # X- matrix of predictors
        # Y- matrix of target variable of shape N x 1
        
        # iter_max- maximum number of iterations 
        # interactions- a matrix with two columns, in which each row specifies a pair of variables between which we want to consider interactions
        # l2- ridge regularization strength
        Y = Y.reshape(-1,1)
        X=np.c_[np.ones(X.shape[0]) , X]
        self.interactions = interactions
        for interaction in interactions:
            X=np.c_[X, X[:,interaction[0]+1] * X[:,interaction[1]+1]]
        β_new = np.transpose([[np.random.uniform() for _ in range(X.shape[1])]]) #Beta initialisation
        i=0
        
        while i==0 or (i < iter_max and (np.sum(np.abs(β_new-β_old))>1e-4)): #Stopping if betas don't change or reached iteration limit
            i+=1
            β_old = β_new
            π = self.sigmoid(X, β_old)
            W = [1/4 for i in range(X.shape[0])] if self.maximisation_minorisation else np.transpose((π*(1-π)))[0]
            
            #Stopping when W approaches singularity
            if 1-self.maximisation_minorisation and np.min(W)<1e-20: 
                break
            
            W = np.diag(W)
            H = np.transpose(X) @ W @ X
            grad = np.transpose(X) @ (Y-π)
            β_new = β_old - np.linalg.inv(H + l2*np.eye(H.shape[0])) @ grad
        self.beta = β_new
        if i==iter_max:
            print("Algorithm didn't converge! Iteration limit reached!")

    def predict(self, X, p=1/2):
        X_exp = np.c_[np.ones(X.shape[0]) , X]
        for interaction in self.interactions:
            X_exp=np.c_[X_exp, X_exp[:,interaction[0]+1] * X_exp[:,interaction[1]+1]]
        return (self.sigmoid(X_exp,self.beta)<p).astype(int)
    
