import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler




class Engine():

    def __init__(self):
        self.nc = 0
        self.d = 0
        self.n_train = 0
        self.classes = {}
        self.scaler = StandardScaler()


    # Function to perform Exploratory Data Analysis on the data frame that is passed to it.
    def performDataAnalysis(self, dataFrame):

        # Extracting the Features from the pandas Dataframe
        X_train = dataFrame.drop("Class", axis=1).to_numpy()   
        Y_train = dataFrame["Class"].to_numpy()

        # Finding the number of unique labels in the Target (T)
        classes = np.unique(Y_train, axis=None)
        
        print(f"$$ DATA ANALYSIS OF THE UCI WINE DATAS $$")
        print("-------------------------------------------------------------------")
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Number of Features in the Training Data: {X_train.shape[1]}")
        print(f"Number of Data-Points in the Training Data: {X_train.shape[0]}")
        print(f"Number of classes in Y_train is: {len(classes)}")
        print(f"{dict(zip(['Barolo wine','Grignolino wine','Barbera wine'], [1,2,3]))}")
        print("-------------------------------------------------------------------")


    # User-defined Function to generate the Train Data.
    def generateTrainData(self, dataFrame):

        # Extracting the Features from the pandas Dataframe
        X_train = dataFrame.drop("Class", axis=1).to_numpy()   
        Y_train = dataFrame["Class"].to_numpy()

        # Finding the number of unique labels in the Target (T)
        classes = np.unique(Y_train, axis=None)
        
        self.nc = len(classes)
        self.classes = dict(zip(['Barolo wine','Grignolino wine','Barbera wine'], [1,2,3]))
        self.n_train = X_train.shape[0]
        self.d = X_train.shape[1]

        print("Generated the Train Data!")

        return (self.n_train, X_train, Y_train)
    

    # User-defined Function to Standardize the numpy Data passed.
    def standardizeData(self, X):

        '''
            Preprocessing the train data using Scikit-learn.
        '''
        
        self.scaler.fit(X)
        X_std = self.scaler.transform(X)

        return X_std
    
    
    



    