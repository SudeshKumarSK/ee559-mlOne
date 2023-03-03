import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class Perceptron():

    def __init__(self):
        self.d = 0 # No of features in the input training data. (test data is also same).
        self.n_train = 0 # No. of data points in the input training data.
        self.n_test = 0 
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


    def generateTestData(self, test_data, printFlag = False):

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
        self.n_test = test_data_np.shape[0]

        if printFlag:
            print("---------------------------------------------------")
            print(f"  Shape of Input Data: {test_data_np.shape}")
            print(f"  Number of Data Points: {self.n_test}")
            print(f"  Number of Input Features: {self.d}")
            print(f"  Number of Target Classes: {self.nc}")
            print("---------------------------------------------------")
        return (self.n_test, X_test, T_test)
    
    def standardizeTrainData(self, X_train):

        '''
            Preprocessing the train data using Scikit-learn.
        '''
        
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train_std = self.scaler.transform(X_train)

        return X_train_std


    def standardizeTestData(self, X_test):

        '''
            Preprocessing the train data using Scikit-learn.
        '''

        X_test_std = self.scaler.transform(X_test)

        return X_test_std

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
        # np.random.seed(42)

        # Shuffle the combined array
        np.random.shuffle(combinedData)

        # Split the shuffled array back into data points and labels
        shuffled_data_points = combinedData[:, 0:-1]
        shuffled_labels = combinedData[:, -1:]

        return (shuffled_data_points, shuffled_labels)

    

    def computeCost(self, X_train, T_train, n_train, w_vector):
        J = 0

        for i in range(n_train):
            curr_loss = (w_vector.T @ X_train[i].reshape(self.d+1, 1)) * T_train[i]
            curr_loss = np.squeeze(curr_loss)

            if curr_loss < 0:
                J = J + (curr_loss)

        return -1*J


    
    def modelTrain_SGD(self, X_train, T_train, n_train, w_vector, epochs = 100, learn_rate = 1, printFlag = True):

        final_w_vector = np.ones((self.d + 1, 1))
    
        epoch = 0
        cost = float("inf")
        
        while epoch < epochs:
            X_train, T_train = self.shuffleData(X_train, T_train.reshape(n_train, 1))
            for n in range(n_train):

                curr_loss = w_vector.T @ X_train[n].reshape(self.d+1, 1) * T_train[n]
                curr_loss = np.squeeze(curr_loss)
    
                if curr_loss < 0:
                    w_vector = w_vector - learn_rate*(-1 * X_train[n].reshape(self.d+1, 1) * T_train[n])

               
                else:
                    w_vector = w_vector


                if epoch == 99 and n >= self.n_train - 100:
                    current_cost = self.computeCost(w_vector = w_vector, X_train = X_train, T_train = T_train, n_train=n_train)
                    
                    if current_cost < cost:
                        cost = current_cost
                        final_w_vector = w_vector

                        if printFlag:
                            print(f"Epoch #{epoch+1} and Interation #{n} --> Cost J(W) is: {current_cost}")
                            
            if printFlag:
                print(f"Epoch #{epoch+1} --> Cost J(W) is: {self.computeCost(w_vector = w_vector, X_train = X_train, T_train = T_train, n_train=n_train)}")

            epoch += 1
    
        return final_w_vector



    def predict(self, optimum_w_vector, X):

        Y_hat = np.dot(optimum_w_vector.T, X.T).T

        Y_hat = np.sign(Y_hat)


        return Y_hat
    

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
    

    def updateDimension(self, d):
        self.d = d


    
    
