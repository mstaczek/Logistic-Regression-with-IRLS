import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LogisticRegressionIRLS import LR
#### Data generation
def generate_data(B = [0, 0, 0, 1], n = 1000, seed=123, std = 1/10):
    
    np.random.seed(seed)
    
    X1 = np.random.normal(0,1,n)
    X2 = np.random.normal(0,1,n)
    X3 = X1*X2
    
    X = pd.DataFrame(np.transpose([X1, X2]))
    Y =  np.transpose((1/(1+np.exp([B] @ np.array([[1 for i in range(n)], X1, X2, X3]+np.random.normal(0,std,n))))>1/2).astype(int))
    return X, Y


#Main method used for running and visualisin redults of experiments
def run_experiment(B = [0, 0, 0, 1], interactions = False, title=None, seed=123):
    X, Y = generate_data(B, seed=seed)

    lr = LR()
    if interactions:
        lr.fit(X,Y, interactions=[[0,1]], l2=0)
    else:
        lr.fit(X,Y, l2=0)

    y_pred =  lr.predict(X)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].scatter(X[0], X[1], c=Y, alpha=0.5, s=40)
    ax[0].set_title('True labels')
    ax[1].scatter(X[0], X[1], c=y_pred, alpha=0.5, s=40)
    ax[1].set_title('Predicted labels')
    ax[2].scatter(X[0], X[1], c=-np.abs(y_pred-Y), cmap='RdYlGn', s=40, alpha=0.5)
    ax[2].set_title('Difference')
    accuracy = 1-np.mean(np.abs(y_pred-Y))
    if title:
        fig.suptitle(title + f', accuracy: {accuracy:.2%}')
    else:
        fig.suptitle(f'Accuracy: {accuracy:.2%}')    

    plt.show()
    
def add_decision_boundary(ax, lr):
    # add decision boundary
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    x1, x2 = np.meshgrid(x1, x2)
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    X_exp = np.c_[np.ones(x1.shape[0]) , x1, x2]
    for interaction in lr.interactions:
        X_exp=np.c_[X_exp, X_exp[:,interaction[0]+1] * X_exp[:,interaction[1]+1]]
    x1 = x1.reshape(100,100)
    x2 = x2.reshape(100,100)
    x3 = (lr.sigmoid(X_exp, lr.beta)>0.5).astype(int).reshape(100,100)

    ax[1].contour(x1, x2, x3, levels=[0.5], colors='k', linestyles='solid')