import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score





class Engine():

    def __init__(self):
        self.nc = 0
        self.d = 0
        self.n_train = 0
        self.classes = {}
        self.scaler = StandardScaler()

        # Initializing the KFold as 4 and creating an instance for sklearn's KFold  
        self.kf = KFold(n_splits=20)

        self.X_train = 0
        self.Y_train = 0

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
        print(f"{dict(zip(['Barolo wine - 1','Grignolino wine - 2','Barbera wine - 3'], [1,2,3]))}")
        print("-------------------------------------------------------------------")


    # User-defined Function to generate the Train Data.
    def generateTrainData(self, dataFrame):

        # Extracting the Features from the pandas Dataframe
        self.X_train = dataFrame.drop("Class", axis=1).to_numpy()   
        self.Y_train = dataFrame["Class"].to_numpy()

        # Finding the number of unique labels in the Target (T)
        classes = np.unique(self.Y_train, axis=None)
        
        self.nc = len(classes)
        self.classes = dict(zip([1,2,3], ['Barolo wine - 1','Grignolino wine - 2','Barbera wine - 3']))
        self.n_train = self.X_train.shape[0]
        self.d = self.X_train.shape[1]

        print("Generated the Train Data!")

        return (self.n_train, self.X_train, self.Y_train)
    

    # User-defined Function to Standardize the numpy Data passed.
    def standardizeData(self, X):

        '''
            Preprocessing the train data using Scikit-learn.
        '''
        
        self.scaler.fit(X)
        X_std = self.scaler.transform(X)
        print("Standardized the Train Data!")

        return X_std
    

    
    """ def performProjection(self, X_train, feature1, feature2):

        B = np.stack((X_train[:, feature1 - 1], X_train[:, feature2 - 1]), axis = 0)
        print(B.shape)

        # Compute the Projection Matrix.
        proj = np.dot(X_train.T, B.T) / np.linalg.norm(B)**2

        proj_Mat = np.dot(B.T, B)

        projected_data = np.dot(X_train, proj_Mat)

        return proj
          
        """


    
    def plotDataReducedFeatures(self, X_train, Y_train, feature1, feature2, position = "upper right"):
        # Create a new figure with larger size
        fig = plt.figure(figsize=(6, 6))

        plt.scatter(X_train[Y_train == 1, feature1 - 1], X_train[Y_train == 1, feature2 - 1], alpha = 0.85, c = "#FF6D60", marker = "o", label = f"{self.classes[1]}")
        plt.scatter(X_train[Y_train == 2, feature1 - 1], X_train[Y_train == 2, feature2 - 1], alpha = 0.85, c = "#569DAA", marker = "x", label = f"{self.classes[2]}")
        plt.scatter(X_train[Y_train == 3, feature1 - 1], X_train[Y_train == 3, feature2 - 1], alpha = 0.85, c = "#263A29", marker = "^", label = f"{self.classes[3]}")

        plt.title(f"Plot of Features {feature1} and {feature2}")
        plt.legend(loc = position)
        plt.xlabel(f"Feature x{feature1}")
        plt.ylabel(f"Feature x{feature2}")
        plt.show()

    
    def shuffleData(self, n, X, Y, feature1, feature2):
        
        B = np.stack((X[:, 1 - 1], X[:, 2 - 1]), axis = 0).T
        combinedData = np.hstack((B, Y.reshape(n, 1)))
    
        # Set seed for reproducibility
        # np.random.seed(42)

        # Shuffle the combined array
        np.random.shuffle(combinedData)

        # Split the shuffled array back into data points and labels
        shuffled_data_points = combinedData[:, 0:-1]
        shuffled_labels = combinedData[:, -1:]
        shuffled_labels = shuffled_labels.astype((int))

        return (shuffled_data_points, shuffled_labels.reshape(n,))
    


    def train_MCP(self, n_train, X_train, Y_train, feature1, feature2, runs = 5):
        model_Dict = {}
        run_val_CER = []

        run_mean_val_CER = 0
        run_std_val_CER = 0

        run_minimum_val_CER = 0
        run_maximum_val_CER = 0

        run_max = 0
        run_min = 0

        for run in range(runs):

            val_CER = []
            mean_val_CER = 0

            X_train_shuffled, Y_train_shuffled = self.shuffleData(n_train, X_train, Y_train, feature1, feature2)

            for i, (train_index, val_index) in enumerate(self.kf.split(X_train_shuffled)):

                # Getting the folds of train and validation data one by one 20 times in this loop.
                # Basically X_train_fold will contain n_train/20 data-points in one iteration and X_val_fold will had the rest 1000 data-points.
                X_train_fold, X_val_fold = X_train_shuffled[train_index], X_train_shuffled[val_index]
                Y_train_fold, Y_val_fold = Y_train_shuffled[train_index], Y_train_shuffled[val_index]

                clf = Perceptron(tol=1e-3, random_state=0)

                model = clf.fit(X_train_fold, Y_train_fold)

                if i == 0:
                    model_Dict[run] = model

                # Performing the same steps for the validation fold.
                Y_pred_val_fold = clf.predict(X_val_fold)
                
                val_acc = accuracy_score(Y_pred_val_fold, Y_val_fold)
                cer = 1 - val_acc

                val_CER.append(cer)

            mean_val_CER = np.mean(val_CER)
            run_val_CER.append(mean_val_CER)

        run_mean_val_CER = np.mean(run_val_CER)
        run_std_val_CER = np.std(run_val_CER)

        run_maximum_val_CER = np.max(run_val_CER)
        run_minimum_val_CER = np.min(run_val_CER)

        run_max = np.argmax(run_val_CER)
        run_min = np.argmin(run_val_CER)

        print("-----------------------------------------------------------------------------------------------------------------------------------")
        print(f"The Mean Classification Error Rate from each cross-val run are: {dict(zip([1, 2, 3, 4, 5], run_val_CER))}")
        print(f"The Average of the Mean Classification Error Rate over the 5 runs is: {run_mean_val_CER}")
        print(f"The Standard Deviation of the Mean Classification Error Rate over the 5 runs is: {run_std_val_CER}")
        print(f"The Lowest Mean Classification Error Rate -> {run_minimum_val_CER} was achieved at run: {run_min + 1}")
        print(f"The Higesh Mean Classification Error Rate -> {run_maximum_val_CER} was achieved at run: {run_max + 1}")
        print("-----------------------------------------------------------------------------------------------------------------------------------")


        return (run_max, run_min, model_Dict)
    
    

        






        


    
















    



    