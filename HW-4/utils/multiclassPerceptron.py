# Importing all necessary libraries
import numpy as np
from sklearn.preprocessing import StandardScaler



# Defining the class for MultiClassPerceptron
class MultiClassPerceptron:

    def __init__(self):
        self.d = 0
        self.n_train = 0
        self.n_test = 0
        self.nc = 0 # No. of classes in the training and test data. (Only 2 for perceptron)
        self.classes = {} # classes hold the different labels in the target(T).
        self.scaler = 0
    

    def generateTrainData(self, trainData, printFlag = True):
        
        self.d = trainData.shape[1] - 1

        self.n_train = trainData.shape[0]

        # Converting the sorted dataframe to numpy array.
        train_data_np = trainData.to_numpy()

        # Spliting the data_np into input features (X) and output target (T) basically splitting labels and features.
        X_train = trainData.drop("Class", axis=1).to_numpy()
        T_train = trainData['Class'].replace(['SEKER','DERMASON','BOMBAY','HOROZ','CALI','SIRA','BARBUNYA'],[0,1,2,3,4,5,6]).to_numpy()

        # Finding the number of unique labels in the Target (T)
        classes = np.unique(T_train, axis=None)

        self.nc = len(classes)

        self.classes = dict(zip(['SEKER','DERMASON','BOMBAY','HOROZ','CALI','SIRA','BARBUNYA'], [0,1,2,3,4,5,6]))

        if printFlag:
            print("---------------------------------------------------")
            print(f"TRAIN DATA OF BEAN DATASET: ")
            print(f"  Shape of Input Data: {train_data_np.shape}")
            print(f"  Number of Data Points: {self.n_train}")
            print(f"  Number of Input Features: {self.d}")
            print(f"  Number of Target Classes: {self.nc}")
            print("---------------------------------------------------")


        return (self.n_train, X_train, T_train)


    def generateTestData(self, testData, printFlag = True):

        self.n_test = testData.shape[0]

        test_data_np = testData.to_numpy()

        # Spliting the data_np into input features (X) and output target (T) basically splitting labels and features.
        X_test = testData.drop("Class", axis=1).to_numpy()
        T_test = testData['Class'].replace(['SEKER','DERMASON','BOMBAY','HOROZ','CALI','SIRA','BARBUNYA'],[0,1,2,3,4,5,6]).to_numpy()


        if printFlag:
            print("---------------------------------------------------")
            print(f"TEST DATA OF BEAN DATASET: ")
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
    

    def shuffleData(self, X, T):
        '''
        
        ''' 

        indices = np.random.permutation(X.shape[0])

        shuffled_data_points = X[indices]
        shuffled_labels = T[indices]

        return (shuffled_data_points, shuffled_labels)
    

    def initializeWeights(self, a):

        '''
        Generates the weights(w) and bias(w0) as a whole numpy array based on the number of features in the training data, d.

        Input -> a which is used to alter the values of the initial weights.
        Output -> numpy array of weights(w) and bias (w0) as a whole, w_vector.
        
        '''

        # Dimension of w_vector is # of features x # of classes.
        w_vector = np.ones((self.d+1, self.nc))*a

        return w_vector
    

    def computeCost(self, w_vector, X_train, T_train):
        J = 0

        for n in range(self.n_train):
            k = T_train[n]
            g_x = np.dot(w_vector.T, X_train[n])
            l = np.argmax(g_x)

            if k != l:
                # J += np.dot(w_vector[:, k], X_train[n]) - np.dot(w_vector[:, l], X_train[n])
                J += (np.dot(w_vector[:,k], X_train[n]) - np.dot(w_vector[:,l], X_train[n]))

        J = -1 * J

        return J
    

    def predict(self, X, w_vector):
        return np.argmax(np.dot(X, w_vector), axis=1)
    

    def modelTrain_SGD(self, n_train, X_train, T_train, w_vector, epochs=100, learn_rate = 1, printFlag = False):

        final_w_vector = np.ones((self.d + 1, self.nc))
        cost = float("inf")

        for epoch in range(epochs): # halting condition after 100 epochs
            X_train_shuffled, T_train_shuffled = self.shuffleData(X_train, T_train) # Shuffling data before each epoch
            for n in range(n_train):
                # g_x = np.dot(w_vector.T, X_train_shuffled[n])

                g_x = w_vector.T @ X_train_shuffled[n].reshape(self.d+1, 1)

                # Predicted label
                l = np.argmax(g_x) 

                # True Label
                k = T_train_shuffled[n]           

                # if There is a misclassification
                if k != l: 
                    w_vector[:, k] = w_vector[:, k] + learn_rate * X_train_shuffled[n]
                    w_vector[:, l] = w_vector[:, l] - learn_rate * X_train_shuffled[n]
                    
                if epoch == 99 and n >= self.n_train - 100:
                    current_cost = self.computeCost(w_vector = w_vector, X_train = X_train, T_train = T_train)
                    
                    if current_cost < cost:
                        cost = current_cost
                        if printFlag:
                            print(f"Epoch #{epoch+1} and Interation #{n} --> Cost J(W) is: {current_cost}")
                        final_w_vector = w_vector

            if(printFlag):
                print(f"Epoch #{epoch+1} --> Cost J(W) is: {self.computeCost(w_vector = w_vector, X_train = X_train, T_train = T_train)}")

        return final_w_vector



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
    

    def print_weights(self, w_vector):
        '''
        w - numpy of size (num_features,num_classes) - each column corresponds to weight vector of one class
        '''
        column_magnitudes = np.linalg.norm(w_vector, axis=0)
        print("########################################################################")
        for i in range(w_vector.shape[1]):
            print(f"w_vector[{i+1}]")
            print("Magnitude :",column_magnitudes[i])
            print(w_vector[:,i])
            print("########################################################################")




