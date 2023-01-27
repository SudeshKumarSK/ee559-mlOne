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



# The main NearestMeansClassifier class which has member functions to perform:  
#                   1. Generation of numpy data from the input dataframe.
#                   2. Find the # of data points, # of features and # of classes.
#                   3. Calculation of samples_means for the given data.
#                   4. Generate the target labels from the input dataframe.
#                   5. Plotting the Data Points, Sample Means, Decision Boundaries and Decision Regions.
#                   6. Perform Nearest Means Classification on the Input Data.
#                   7. Find the Accuracy of the Classification error rate (Accuracy).

class NearestMeansClassifier():

    def __init__(self, data):   
        self.data = data 
        self.X = np.zeros((1,1))
        self.d = 0
        self.n = 0

        self.T = np.zeros((1,))
        self.nc = 0
        self.classes = np.zeros((1,))

        self.sample_means = np.zeros((1,1))
        self.Y_hat = []
        

    def printPattern(self):
        print()
        print("----------#################################################################-------------")
        print()

    

    def generate_data(self):
        n_cols = self.data.shape[1]
        self.d = n_cols-1
        data_sorted = self.data.sort_values(by=self.data.columns[-1])
        data_np = data_sorted.to_numpy()

        
        self.X = data_np[:, 0:n_cols-1]
        self.T = data_np[:, -1]
        self.n = data_np.shape[0]

        classes, class_index, class_count = np.unique(self.T, return_index=True, return_counts=True, axis=None)
        self.classes = classes
        self.nc = len(classes)

        self.sample_means = np.zeros((self.nc, self.d))
        for i in range(self.nc - 1):
            self.sample_means[i] = np.mean(self.X[class_index[i] : class_index[i+1]], axis=0)

        self.sample_means[self.nc-1] = np.mean(self.X[class_index[self.nc - 1]:], axis=0)


        print("---------------------------------------------------")
        print(f"  Shape of Training Data: {data_np.shape}")
        print(f"  Number of Data Points: {self.n}")
        print(f"  Number of Input Features: {self.d}")
        print(f"  Number of Target Classes: {self.nc}")
        print("---------------------------------------------------")
        
        self.printPattern()

        print("---------------------------------------------------")
        print(f"  Shape of sample_means: {self.sample_means.shape}")
        print(f"  Sample Means: ")
        print(self.sample_means)
        print("---------------------------------------------------")

        self.printPattern()



    def findClassMeans(self):
        return self.sample_means


    def plotBoundary(self):
        self.plotDecBoundaries(self.X, self.T, self.sample_means)


    def plotDecBoundaries(self, training, label_train, sample_mean, fsize=(18,18)):
    
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


    def classify(self):
        self.Y_hat = []
        for x in self.X:
            distances = []
            for mean in self.sample_means:
                euclidean_Dist = np.linalg.norm(x - mean)
                distances.append(euclidean_Dist)
            self.Y_hat.append(self.classes[np.argmin(distances)])



    def calculateAccuracy(self, percentageFlag = False):
        totalPredictions = self.n
        correctPredictions = 0

        for i in range(len(self.Y_hat)):
            if self.T[i] == self.Y_hat[i]:
                correctPredictions += 1

        if percentageFlag:
            return (correctPredictions / totalPredictions) * 100
        return correctPredictions / totalPredictions

        

    
        

