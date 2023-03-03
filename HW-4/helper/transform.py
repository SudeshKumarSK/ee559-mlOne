import numpy as np

def quadraticTransformation(X):
    N, D = X.shape
    X_Q = np.zeros((N,D*(D+1)//2))
    col_num = 0
    for i in range(D):
        for j in range(i+1):
           X_Q[:,col_num] = X[:,i]*X[:,j]
           col_num+=1
    return X_Q


def cubicTransformation(X):
    N, D = X.shape
    X_C = np.zeros((N,D*(D+1)*(D+2)//6))
    col_num = 0
    for i in range(D):
        for j in range(i+1):
            for k in range(j+1):
                X_C[:,col_num] = X[:,i]*X[:,j]*X[:,k]
                col_num+=1
    return X_C