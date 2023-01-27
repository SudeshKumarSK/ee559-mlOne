################################################
## EE559 HW-1.
## Created by Sudesh Kumar Santhosh Kumar.
## Tested in Python 3.10.9 using conda environment version 22.9.0.
################################################

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist



class NearestMeansClassifier():

    '''
        The main NearestMeansClassifier class which has member functions to perform:  

                   1. Generation of numpy data from the input dataframe.
                   2. Find the # of data points, # of features and # of classes.
                   3. Calculation of samples_means for the given data.
                   4. Generate the target labels from the input dataframe.
                   5. Plotting the Data Points, Sample Means, Decision Boundaries and Decision Regions.
                   6. Perform Nearest Means Classification on the Input Data.
                   7. Find the Accuracy and Classification Error Rate of the Classifier.

    '''

    def __init__(self):   

        '''
            Non-Paramertised Constructor for the NearestMeansClassifier clas.
        '''

        # All data-members of the class are declared and initialized below.

        self.d = 0  # d is the number of features in the input data(X).
        self.n = 0  # n is the number of data points in the input data(X).
        self.nc = 0 # nc is the number of classes in the target(t).
        self.classes = np.zeros((1,)) # classes hold the different labels in the target(T).
        self.classIndices = np.zeros((1,)) # classIndices hold the index of where a labels starts in the sorted data (X)
        self.sample_means = np.zeros((1,1)) # sample_means holds the sample_means calculated from the input features.
    


    def generateData(self, data):

        '''
        Generate numpy array of input data points, X and Target vector, T


        input -> Pandas dataframe.
        output -> tuple of input features (X), # of data points in X, Target labels (X, d, T)

        '''

        # n_cols = data.shape[1]
        # Storing the # of features in self.d calculated from the shape of input dataframe.
        self.d = data.shape[1] - 1
        
        # Storing the # of data points in self.n calculated from the shape of input dataframe.
        self.n = data.shape[0]

        # Sorting the input dataframe based on the target labels and generating a new dataframe data_sorted.
        data_sorted = data.sort_values(by=data.columns[-1])

        # Converting the sorted dataframe to numpy array.
        data_np = data_sorted.to_numpy()

        # Spliting the data_np into input features (X) and output target (T) basically splitting labels and features.
        X = data_np[:, 0:self.d]
        T = data_np[:, -1]
        
        # Finding the number of unique labels in the Target (T)
        classes, class_index, class_count = np.unique(T, return_index=True, return_counts=True, axis=None)

        # Storing the different types of classes (lables) in self.classes. For Eg, it can either be (0, 1) or (1, 2) or (1, 2, 3).
        self.classes = classes

        # Storing where the different labels start in the target (T). We will use this to compute the class means (or) sample means.
        self.classIndices = class_index

        # Storing the # of unique classes in self.nc calculated from the classes returned by np.unique().
        self.nc = len(classes)


        print("---------------------------------------------------")
        print(f"  Shape of Input Data: {data_np.shape}")
        print(f"  Number of Data Points: {self.n}")
        print(f"  Number of Input Features: {self.d}")
        print(f"  Number of Target Classes: {self.nc}")
        print("---------------------------------------------------")
        
        return (X, self.n, T)

    def transformTestData(self, test_data):

        '''
        Transforms the test_Data pandas dataframe into numpy array and splits it into features and true labels.
        
        Input -> test_data which is a pandas dataframe of testing data.
        Output -> Tuple of input features of test data, # of features in the input data, True labels vector for test data
        (X_test, n_test, T_test).
        '''

        # Converting the test dataframe to numpy array.
        test_data_np = test_data.to_numpy()

        # Spliting the test_data_np into input features (X_test) and true labels vector (T_test) 
        # basically splitting labels and features of the test_data.
        X_test = test_data_np[:, 0:self.d]
        T_test = test_data_np[:, -1]
        n_test = test_data_np.shape[0]

        return (X_test, n_test, T_test)


    def calculateClassMeans(self, X):

        '''
        Calculate class means or sample means from the input data points, X

        input -> numpy array of data points (X).
        output -> numpy array, sample_means of shape (nc, d).
        
        '''

        # Re-Initializing self.sample_means with zeros of shape (nc, d) [no. of classes x no. of features]
        self.sample_means = np.zeros((self.nc, self.d))

        for i in range(self.nc - 1):
            self.sample_means[i] = np.mean(X[self.classIndices[i] : self.classIndices[i+1]], axis=0)

        self.sample_means[self.nc-1] = np.mean(X[self.classIndices[self.nc - 1]:], axis=0)

        return self.sample_means
        
 
    def standardizeData(self, X):

        '''
        Standardize the input numpy array of data points X by subtracting mean from every data point and dividing
        every data point by the standard deviation.

        input -> numpy array of data points, X.
        output -> numpy array of standardized data points, X.
        '''
        # Calculating mean for the input features column-wise.
        X_mean = np.mean(X, axis = 0)

        # Calculating standard deviation for the input features column-wise.
        X_std = np.std(X, axis = 0)


        print(f"Shape of X_mean: {X_mean.shape}")
        print(f"Shape of X_std: {X_std.shape}")
        print(f"Mean of X along columns is: {X_mean}")
        print(f"Standard Deviation of X along columns is: {X_std}")

        # Subtract Mean from X (Brodcasting Takes place internally)
        # We can directly subtract mean from X
        X = X - X_mean

        # Divide X by X_std(Brodcasting Takes place internally)
        # We can directly divide X by X_std
        X = X / X_std

        print("Input Data, X has been Standardized successfully!")

        return X


    
    def plotDecisionBoundaries(self, training, label_train, sample_mean, fsize=(18,18)):
    
        '''
        Plot the decision boundaries and data points for minimum distance to
        class mean classifier
        
        training: traning data, N x d matrix:
            N: number of data points
            d: number of features
            if d > 2 then the first and second features will be plotted (1st and 2nd column (0 and 1 index))
        label_train: class lables correspond to training data, N x 1 array:
            N: number of data points
            the labels should start numbering from 1 (not 0)
            code works for up to 3 classes
        sample_mean: mean vector for each class, C x d matrix:
            C: number of classes
            each row of the sample_mean matrix is the coordinate of each sample mean
        '''

        #
        # Total number of classes
        nclass =  max(np.unique(label_train))

        # Set the feature range for ploting
        max_x = np.ceil(max(training[:, 0])) + 1
        min_x = np.floor(min(training[:, 0])) - 1
        max_y = np.ceil(max(training[:, 1])) + 1
        min_y = np.floor(min(training[:, 1])) - 1

        xrange = (min_x, max_x)
        yrange = (min_y, max_y)

        # step size for how finely you want to visualize the decision boundary.
        inc = 0.005

        # generate grid coordinates. this will be the basis of the decision
        # boundary visualization.
        (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

        # size of the (x, y) image, which will also be the size of the
        # decision boundary image that is used as the plot background.
        image_size = x.shape
        xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) # make (x,y) pairs as a bunch of row vectors.

        # distance measure evaluations for each (x,y) pair.
        dist_mat = cdist(xy, sample_mean)
        pred_label = np.argmin(dist_mat, axis=1)

        # reshape the idx (which contains the class label) into an image.
        decisionmap = pred_label.reshape(image_size, order='F')



        #show the image, give each coordinate a color according to its class label
        plt.figure(figsize=fsize)
        plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

        # plot the class training data.
        plt.plot(training[label_train == 1, 0],training[label_train == 1, 1], 'rx')
        plt.plot(training[label_train == 2, 0],training[label_train == 2, 1], 'go')
        if nclass == 3:
            plt.plot(training[label_train == 3, 0],training[label_train == 3, 1], 'b*')

        # include legend for training data
        if nclass == 3:
            l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
        else:
            l = plt.legend(('Class 1', 'Class 2'), loc=2)
        plt.gca().add_artist(l)

        # plot the class mean vector.
        m1, = plt.plot(sample_mean[0,0], sample_mean[0,1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
        m2, = plt.plot(sample_mean[1,0], sample_mean[1,1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
        if nclass == 3:
            m3, = plt.plot(sample_mean[2,0], sample_mean[2,1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')

        # include legend for class mean vector
        if nclass == 3:
            l1 = plt.legend([m1,m2,m3],['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)
        else:
            l1 = plt.legend([m1,m2], ['Class 1 Mean', 'Class 2 Mean'], loc=4)

        plt.gca().add_artist(l1)

        plt.show()



    def classify(self, X):

        '''
        It takes the input numpy array of data points and predicts the labels f
        or the data points and returns a list of predictions.

        input -> numpy array of data points, X.
        output -> list of predicted output labels, Y_hat.

        '''

        # List to store the predictions.
        Y_hat = []

        # Iterate over every data point in the input X
        for x in X:

            # distances stores the euclidean distance of the input data point from sample_means.
            # distance will have a length same as the # of classes (nc )
            distances = []

            for mean in self.sample_means:
                # If a data point is made up of 2 features then x.shape will be (2,).
                euclidean_Dist = np.linalg.norm(x - mean)
                distances.append(euclidean_Dist)
            Y_hat.append(self.classes[np.argmin(distances)])
        
        return Y_hat


    def calculateCER(self, T, Y_hat, percentageFlag = False):

        '''
        Calculates the Classification Error Rate from the Target Vector(T) and Prediction Vector(Y_hat)

        Input -> Target Vector(T), Prediction Vector(Y_hat) and a percentageFlag to display CER in percentage or not.
        Output -> Classification Error Rate in float or percentage value.
        
        '''
        totalPredictions = self.n
        incorrectPredictions = 0

        for i in range(len(Y_hat)):
            if T[i] != Y_hat[i]:
                incorrectPredictions += 1

        if percentageFlag:
            return (incorrectPredictions / totalPredictions) * 100
        return incorrectPredictions / totalPredictions



    def calculateAccuracy(self, T, Y_hat, percentageFlag = False):

        '''
        Calculates the Classifier's Accuracy from the Target Vector(T) and Prediction Vector(Y_hat)

        Input -> Target Vector(T), Prediction Vector(Y_hat) and a percentageFlag to display CER in percentage or not.
        Output -> Accuracy in float or percentage value.
        
        '''

        totalPredictions = self.n
        correctPredictions = 0

        for i in range(len(Y_hat)):
            if T[i] == Y_hat[i]:
                correctPredictions += 1

        if percentageFlag:
            return (correctPredictions / totalPredictions) * 100
        return correctPredictions / totalPredictions