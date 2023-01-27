import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def printPattern():
    print()
    print("----------#################################################################-------------")
    print()

def generate_data(train_data):
    n_cols = train_data.shape[1]
    d = n_cols-1
    train_data_sorted = train_data.sort_values(by=train_data.columns[-1])
    train_data_np = train_data_sorted.to_numpy()

    
    x = train_data_np[:, 0:n_cols-1]
    y = train_data_np[:, -1]
    n = train_data_np.shape[0]

    classes, class_index, class_count = np.unique(y, return_index=True, return_counts=True, axis=None)
    nc = len(classes)

    sample_means = np.zeros((nc, d))
    for i in range(nc - 1):
        sample_means[i] = np.mean(x[class_index[i] : class_index[i+1]], axis=0)

    sample_means[nc-1] = np.mean(x[class_index[nc - 1]:], axis=0)


    print("---------------------------------------------------")
    print(f"  Shape of Training Data: {train_data_np.shape}")
    print(f"  Number of Data Points: {n}")
    print(f"  Number of Input Features: {d}")
    print("---------------------------------------------------")
    
    printPattern()

    print(f"Shape of sample_means: {sample_means.shape}")
    print("Sample Means: ")
    print(sample_means)

    printPattern()
    
    return (x, sample_means, y)
