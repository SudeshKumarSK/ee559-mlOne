################################################
## EE559 HW-3.
## Created by Sudesh Kumar Santhosh Kumar.
## Tested in Python 3.10.9 using conda environment version 22.9.0.
################################################

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class Perceptron():
    '''
        The main Perceptron class which contains all the member functions to perform Binary Classification.

        The Perceptron Engine performs the following,
            1. generating training data and labels from pandas dataframe.
            2. generating augmented training data.
            3. changing the labels of the training data to -1 and 1.
            4. shuffling the training data and labels.
            5. generating the initial weights and bias.
            6. generating test data and test labels from pandas dataframe.
            7. Perform the training process to learn the parameters.
            8. Predict the labels from optimal weights.
            9. Calculate Accuracy and CER.
    
    '''

    def __init__(self):
        self.d = 0 # No of features in the input training data. (test data is also same).
        self.n_train = 0 # No. of data points in the input training data.
        self.nc = 0 # No. of classes in the training and test data. (Only 2 for perceptron)
        self.classes = np.zeros((1,)) # classes hold the different labels in the target(T).
        self.classIndices = np.zeros((1,)) # classIndices hold the index of where a labels starts in the sorted data (X)


    def generateTrainData(self, trainData, printFlag = True):

        '''

        Generate numpy array of input data points, X_train and Target vector, T_train.

        input -> Pandas dataframe of trainData.
        output -> tuple of input features (X_train), # of data points in X_train, Target labels (T_train). 

        (X_train, n_train, T_train)

        '''

        # n_cols = data.shape[1]
        # Storing the # of features in self.d calculated from the shape of input dataframe.
        self.d = trainData.shape[1] - 1
        
        # Storing the # of data points in self.n_train calculated from the shape of input dataframe.
        self.n_train = trainData.shape[0]

        # Sorting the input dataframe based on the target labels and generating a new dataframe data_sorted.
        # data_sorted = trainData.sort_values(by=trainData.columns[-1]) 

        # Converting the sorted dataframe to numpy array.
        train_data_np = trainData.to_numpy()

        # Spliting the data_np into input features (X) and output target (T) basically splitting labels and features.
        X_train = train_data_np[:, 0:self.d]
        T_train = train_data_np[:, -1]

        # Finding the number of unique labels in the Target (T)
        classes, class_index, class_count = np.unique(T_train, return_index=True, return_counts=True, axis=None)

        # Storing the different types of classes (lables) in self.classes. For Eg, it can either be (0, 1) or (1, 2) or (1, 2, 3).
        self.classes = classes

        # Storing where the different labels start in the target (T). We will use this to compute the class means (or) sample means.
        self.classIndices = class_index

        # Storing the # of unique classes in self.nc calculated from the classes returned by np.unique().
        self.nc = len(classes)

        if printFlag:
            print("---------------------------------------------------")
            print(f"  Shape of Input Data: {train_data_np.shape}")
            print(f"  Number of Data Points: {self.n_train}")
            print(f"  Number of Input Features: {self.d}")
            print(f"  Number of Target Classes: {self.nc}")
            print("---------------------------------------------------")


        return (self.n_train, X_train, T_train)


    def generateTestData(self, test_data):

        '''
        Transforms the test_data pandas dataframe into numpy array and splits it into features and true labels.

        Input -> test_data which is a pandas dataframe of testing data.
        Output -> Tuple of input features of test data (X_test), # of features in the input data (n_test), 
        True labels vector for test data (T_test)

        (X_test, n_test, T_test).

        '''

        # Converting the test dataframe to numpy array.
        test_data_np = test_data.to_numpy()

        # Spliting the test_data_np into input features (X_test) and true labels vector (T_test) 
        # basically splitting labels and features of the test_data.
        X_test = test_data_np[:, 0:self.d]
        T_test = test_data_np[:, -1]
        n_test = test_data_np.shape[0]

        return (n_test, X_test, T_test)

    def generateTrainDataNumpy(self, trainData, printFlag = True):

        # Storing the # of features in self.d calculated from the shape of input dataframe.
        self.d = trainData.shape[1] - 1
        
        # Storing the # of data points in self.n_train calculated from the shape of input dataframe.
        self.n_train = trainData.shape[0]

        # Sorting the input dataframe based on the target labels and generating a new dataframe data_sorted.
        # data_sorted = trainData.sort_values(by=trainData.columns[-1]) 

        # Spliting the data_np into input features (X) and output target (T) basically splitting labels and features.
        X_train = trainData[:, 1:self.d+1]
        T_train = trainData[:, 0]

        # Finding the number of unique labels in the Target (T)
        classes, class_index, class_count = np.unique(T_train, return_index=True, return_counts=True, axis=None)

        # Storing the different types of classes (lables) in self.classes. For Eg, it can either be (0, 1) or (1, 2) or (1, 2, 3).
        self.classes = classes

        # Storing where the different labels start in the target (T). We will use this to compute the class means (or) sample means.
        self.classIndices = class_index

        # Storing the # of unique classes in self.nc calculated from the classes returned by np.unique().
        self.nc = len(classes)

        if printFlag:
            print("---------------------------------------------------")
            print(f"  Shape of Input Data: {trainData.shape}")
            print(f"  Number of Data Points: {self.n_train}")
            print(f"  Number of Input Features: {self.d}")
            print(f"  Number of Target Classes: {self.nc}")
            print("---------------------------------------------------")


        return (self.n_train, X_train, T_train)


    def generateTestDataNumy(self, test_data):

        '''
        Transforms the test_data pandas dataframe into numpy array and splits it into features and true labels.

        Input -> test_data which is a pandas dataframe of testing data.
        Output -> Tuple of input features of test data (X_test), # of features in the input data (n_test), 
        True labels vector for test data (T_test)

        (X_test, n_test, T_test).

        '''

        # Spliting the test_data_np into input features (X_test) and true labels vector (T_test) 
        # basically splitting labels and features of the test_data.
        X_test = test_data[:, 1:self.d+1]
        T_test = test_data[:, 0]
        n_test = test_data.shape[0]

        return (n_test, X_test, T_test)


    def augmentData(self, X, n):

        '''
        Augments the input numpy data by adding a column of ones which is treated as x0.

        Input -> numpy array of features, X.
        Output -> Augmented numpy array X_augmented.

        '''

        X_augmented = np.hstack((np.ones((n, 1)), X))
        
        return X_augmented


    def changeLabels(self, T):

        '''
        Changes the labels of the input numpy array of labels to -1 and 1. For eg. If T has two classes 1 and 2 it gets
        changed into -1 and 1 respectively. It can also be 0 and 1 or anything.

        Input -> numpy array of labels, T.
        Output -> numpy array of changed labels, T_changed.
        
        '''

        label_1 = self.classes[0]
        label_2 = self.classes[1]

        T_changed = np.copy(T)

        T_changed[T ==label_1] = 1.0
        T_changed[T ==label_2] = -1.0

        return T_changed


    def initializeWeights(self, a):

        '''
        Generates the weights(w) and bias(w0) as a whole numpy array based on the number of features in the training data, d.

        Input -> a which is used to alter the values of the initial weights.
        Output -> numpy array of weights(w) and bias (w0) as a whole, w_vector.
        
        '''
        w_vector = np.ones((self.d+1, 1))*a
        return w_vector


    def shuffleData(self, X, T):

        '''
        
        '''
        combinedData = np.hstack((X, T))
        
        # Set seed for reproducibility
        np.random.seed(42)

        # Shuffle the combined array
        np.random.shuffle(combinedData)

        # Split the shuffled array back into data points and labels
        shuffled_data_points = combinedData[:, 0:-1]
        shuffled_labels = combinedData[:, -1:]

        return (shuffled_data_points, shuffled_labels)

    def computeCost(self, X_train, T_train, n_train, d, w_vector):
        J = 0

        for i in range(n_train):
            curr_loss = (w_vector.T @ X_train[i].reshape(d, 1)) * T_train[i]
            curr_loss = np.squeeze(curr_loss)

            if curr_loss < 0:
                J = J + (curr_loss)

        return -1*J


    def modelTrain_SequentialGD(self, n_train, X_train, T_train, w_vector, epochs = 10, learn_rate = 1):
        '''
        
        '''
        convergenceFlag = False
        J_History_iterations = []
        J_History_epochs= []
        w_History_epochs = []
        cer_History_iterations = []
        cer_History_epochs = []

        
        n_iters_arr = []
        n_epochs_arr = []

        m = 0
        n_iters = 1

        # d = X_train[0].shape[0]
        # J_i = self.computeCost(X_train=X_train, T_train=T_train, n_train=n_train, d=d, w_vector=w_vector)
        # J_History_iterations.append(J_i)
        # J_History_epochs.append(J_i)
        # w_History_epochs.append(w_vector)

        # Y_hat = self.predict(X=X_train, w_optimum=w_vector)
        # curr_cer = self.calculateCER(T=T_train, Y_hat=Y_hat, n=self.n_train)
        # cer_History_iterations.append(curr_cer)
        # cer_History_epochs.append(curr_cer)
       
        while  m < epochs:
            n_epochs_arr.append(m+1)
            J_m = 0

            for n in range(n_train):
                n_iters_arr.append(n_iters)

                d = X_train[n].shape[0]
                
                curr_loss = w_vector.T @ X_train[n].reshape(d, 1) * T_train[n]
                curr_loss = np.squeeze(curr_loss)
    
                if curr_loss < 0:
                    w_vector = w_vector - learn_rate*(-1 * X_train[n].reshape(d, 1) * T_train[n])
                
                J_i = self.computeCost(X_train=X_train, T_train=T_train, n_train=n_train, d=d, w_vector=w_vector)
                J_History_iterations.append(J_i)

                Y_hat = self.predict(X=X_train, w_optimum=w_vector)
                curr_cer = self.calculateCER(T=T_train, Y_hat=Y_hat, n=self.n_train)
                cer_History_iterations.append(curr_cer)

                if curr_cer == 0:
                    convergenceFlag = True
                    break
                
                n_iters += 1

            J_m = J_i
            J_History_epochs.append(J_m)
            w_History_epochs.append(w_vector)
            cer_History_epochs.append(curr_cer)

            if convergenceFlag:
                return (convergenceFlag, m+1, n_epochs_arr, n_iters, n_iters_arr, J_History_epochs, w_History_epochs, J_History_iterations, cer_History_epochs, cer_History_iterations)

            m += 1
  
        return (convergenceFlag, m+1, n_epochs_arr, n_iters, n_iters_arr, J_History_epochs, w_History_epochs, J_History_iterations, cer_History_epochs, cer_History_iterations)


    def modelTrain_Stochastic_GD_Variant1(self, n_train, X_train, T_train, w_vector, epochs = 10, learn_rate = 1):
        '''
        
        '''
        convergenceFlag = False
        J_History_iterations = []
        J_History_epochs= []
        w_History_epochs = []
        cer_History_iterations = []
        cer_History_epochs = []

        
        n_iters_arr = []
        n_epochs_arr = []

        m = 0
        n_iters = 1

        # d = X_train[0].shape[0]
        # J_i = self.computeCost(X_train=X_train, T_train=T_train, n_train=n_train, d=d, w_vector=w_vector)
        # J_History_iterations.append(J_i)
        # J_History_epochs.append(J_i)
        # w_History_epochs.append(w_vector)

        # Y_hat = self.predict(X=X_train, w_optimum=w_vector)
        # curr_cer = self.calculateCER(T=T_train, Y_hat=Y_hat, n=self.n_train)
        # cer_History_iterations.append(curr_cer)
        # cer_History_epochs.append(curr_cer)
       
        while  m < epochs:
            n_epochs_arr.append(m+1)
            J_m = 0

            for n in range(n_train):
                n_iters_arr.append(n_iters)

                d = X_train[n].shape[0]
                
                curr_loss = w_vector.T @ X_train[n].reshape(d, 1) * T_train[n]
                curr_loss = np.squeeze(curr_loss)
    
                if curr_loss < 0:
                    w_vector = w_vector - learn_rate*(-1 * X_train[n].reshape(d, 1) * T_train[n])
                
                J_i = self.computeCost(X_train=X_train, T_train=T_train, n_train=n_train, d=d, w_vector=w_vector)
                J_History_iterations.append(J_i)

                Y_hat = self.predict(X=X_train, w_optimum=w_vector)
                curr_cer = self.calculateCER(T=T_train, Y_hat=Y_hat, n=self.n_train)
                cer_History_iterations.append(curr_cer)

                if curr_cer == 0:
                    convergenceFlag = True
                    break
                
                n_iters += 1

            J_m = J_i
            J_History_epochs.append(J_m)
            w_History_epochs.append(w_vector)
            cer_History_epochs.append(curr_cer)

            if convergenceFlag:
                return (convergenceFlag, m+1, n_epochs_arr, n_iters, n_iters_arr, J_History_epochs, w_History_epochs, J_History_iterations, cer_History_epochs, cer_History_iterations)
            
            X_train, T_train = self.shuffleData(X=X_train, T=T_train)
            m += 1
  
        return (convergenceFlag, m+1, n_epochs_arr, n_iters, n_iters_arr, J_History_epochs, w_History_epochs, J_History_iterations, cer_History_epochs, cer_History_iterations)

    def predict(self, X, w_optimum):

        '''
        Predicts the output labels for the given numpy array of features, X and optimal w_vector learned during the training 
        process.

        Input -> numpy array of features, X and optimal w_vector, w_optimum.
        Output -> numpy array of predicted labels, Y_hat.
        
        '''

        Y_hat = w_optimum.T @ X.T

        Y_hat = Y_hat.T

        Y_hat = np.sign(Y_hat)

        return Y_hat


    def calculateCER(self, T, Y_hat, n):

        '''
        Calculates the Classification Error Rate from the Target Vector(T) and Prediction Vector(Y_hat)

        Input -> Target Vector(T), Prediction Vector(Y_hat) and a percentageFlag to display CER in percentage or not.
        Output -> Classification Error Rate percentage.
        
        '''
        totalPredictions = n
        incorrectPredictions = 0

        for i in range(len(Y_hat)):
            if T[i] != Y_hat[i]:
                incorrectPredictions += 1

        return (incorrectPredictions / totalPredictions) * 100



    def calculateAccuracy(self, T, Y_hat, n):

        '''
        Calculates the Classifier's Accuracy from the Target Vector(T) and Prediction Vector(Y_hat)

        Input -> Target Vector(T), Prediction Vector(Y_hat) and a percentageFlag to display CER in percentage or not.
        Output -> Accuracy in percentage.
        
        '''

        totalPredictions = n
        correctPredictions = 0

        for i in range(len(Y_hat)):
            if T[i] == Y_hat[i]:
                correctPredictions += 1

        return (correctPredictions / totalPredictions) * 100


    def plotCriterionVsIters(self, n_iters, J_History_iters, J_optimum_iters, datasetName):

        ax = plt.axes()
        ax.plot(n_iters, J_History_iters, c = "grey")


        for i, n in enumerate(n_iters):
            if J_History_iters[i] == J_optimum_iters:
                ax.scatter(n, J_History_iters[i], c='purple', marker='o', s = 50, alpha=1)
            else:
                ax.scatter(n, J_History_iters[i], c='orange', marker='o', s = 50, alpha=1)

        ax.set_title("Criterion Function Vs. Number of Iterations " + "(" + datasetName + ")")
        ax.set_ylabel('Criterion Function J(w)')
        ax.set_xlabel('Number of Iterations')

    plt.show()

    def plotCERVsIters(self, n_iters, cer_History_iters, cer_optimum_iters, datasetName):

        ax = plt.axes()
        ax.plot(n_iters, cer_History_iters, c = "grey")


        for i, n in enumerate(n_iters):
            if cer_History_iters[i] == cer_optimum_iters:
                ax.scatter(n, cer_History_iters[i], c='purple', marker='o', s = 50, alpha=1)
            else:
                ax.scatter(n, cer_History_iters[i], c='orange', marker='o', s = 50, alpha=1)

        ax.set_title("Classification Error Rate Vs. Number of Iterations " + "(" + datasetName + ")")
        ax.set_ylabel('Classification Error Rate (CER)')
        ax.set_xlabel('Number of Iterations')

    plt.show()






    def plotCriterionVsEpochs(self, n_epochs, J_History_epochs, J_optimum_epochs, datasetName):

        ax = plt.axes()
        ax.plot(n_epochs, J_History_epochs, c = "grey")


        for i, n in enumerate(n_epochs):
            if J_History_epochs[i] == J_optimum_epochs:
                ax.scatter(n, J_History_epochs[i], c='purple', marker='o', s = 50, alpha=1)
            else:
                ax.scatter(n, J_History_epochs[i], c='orange', marker='o', s = 50, alpha=1)

        ax.set_title("Criterion Function Vs. Number of Epochs " + "(" + datasetName + ")")
        ax.set_ylabel('Criterion Function J(w)')
        ax.set_xlabel('Number of Epochs')

    plt.show()


    def plotCERVsEpochs(self, n_epochs, cer_History_epochs, cer_optimum_epochs, datasetName):

        ax = plt.axes()
        ax.plot(n_epochs, cer_History_epochs, c = "grey")


        for i, n in enumerate(n_epochs):
            if cer_History_epochs[i] == cer_optimum_epochs:
                ax.scatter(n, cer_History_epochs[i], c='purple', marker='o', s = 50, alpha=1)
            else:
                ax.scatter(n, cer_History_epochs[i], c='orange', marker='o', s = 50, alpha=1)

        ax.set_title("Classification Error Rate Vs. Number of Epochs " + "(" + datasetName + ")")
        ax.set_ylabel('Classification Error Rate (CER)')
        ax.set_xlabel('Number of Epochs')

    plt.show()


    def plotDecisionBoundary(self, data, w_vector, datasetName):

        ax = plt.axes()
        features = data[:,0:self.d]
        labels = data[:,-1]

        x_min, x_max = np.ceil(max(features[:, 0])) + 1, np.floor(min(features[:, 0])) - 1
        y_min, y_max = np.ceil(max(features[:, 1])) + 1, np.floor(min(features[:, 1])) - 1

        x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        z = w_vector[0] + w_vector[1] * x + w_vector[2] * y

        ax.contour(x, y, z, [0], c="purple")

        ax.scatter(data[data[:, -1] == 1, 0], data[data[:, -1] == 1, 1])
        ax.scatter(data[data[:, -1] == -1, 0], data[data[:, -1] == -1, 1])

        class1 = data[data[:, -1] == 1, 0:1]

        ax.set_title("Plot of Data Points " + "(" + datasetName + ")")
        ax.set_ylabel('Feature 2 (X2)')
        ax.set_xlabel('Feature 2 (X1)')


        plt.show()


    def plotHistogram(self, X, T, w_vector, n_train):
        w_optimum_l2 = np.linalg.norm(w_vector)

        distances_datapoints = []
        distances_datapoints_class1 = []
        distances_datapoints_class2 = []

        for i in range(n_train):
            curr_dist = w_vector.T @ X[i].T / w_optimum_l2
            distances_datapoints.append(np.squeeze(curr_dist))


        X_train_augmented_bc_class1 = X[T == 1.0]
        for i in range(X_train_augmented_bc_class1.shape[0]):
            curr_dist = w_vector.T @ X_train_augmented_bc_class1[i].T / w_optimum_l2
            distances_datapoints_class1.append(np.squeeze(curr_dist))

        X_train_augmented_bc_class2 = X[T == -1.0]
        for i in range(X_train_augmented_bc_class2.shape[0]):
            curr_dist = w_vector.T @ X_train_augmented_bc_class2[i].T / w_optimum_l2
            distances_datapoints_class2.append(np.squeeze(curr_dist))


        ax = plt.axes()
        ax.hist(distances_datapoints, rwidth=0.95, color="#658864", alpha = 0.90, label = "All Datapoints")
        ax.hist(distances_datapoints_class1, rwidth=0.70, color="#820000", alpha = 0.75, label = "Class 1")
        ax.hist(distances_datapoints_class2, rwidth=0.70, color="#00337C", alpha = 0.75, label = "Class 2")

        ax.set_title("Histogram of G(X) / ||w||")
        ax.set_ylabel('Frequency')
        ax.set_xlabel('G(x) / ||w||')

        ax.legend()
        
        plt.show()

        

        


        