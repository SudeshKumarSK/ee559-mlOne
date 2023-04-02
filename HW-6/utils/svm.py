################################################
# EE559 HW-6.
# Created by Sudesh Kumar Santhosh Kumar.
# Tested in Python 3.10.9 using conda environment version 22.9.0.
################################################

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class SVM():

    def __init__(self, nc=0, n_train=0, n_test=0, d=0):
        self.nc = nc
        self.n_train = n_train
        self.n_test = n_test
        self.d = d

        self.model = 0
        self.w_vector = 0
        self.w_knot = 0
        self.supportVector = 0

    def generateTrainData(self, trainData, printFlag=True):
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

        # Converting the sorted dataframe to numpy array.
        train_data_np = trainData.to_numpy()

        # Spliting the data_np into input features (X) and output target (T) basically splitting labels and features.
        X_train = train_data_np[:, 0:self.d]
        T_train = train_data_np[:, -1]

        # Finding the number of unique labels in the Target (T)
        classes, class_index, class_count = np.unique(
            T_train, return_index=True, return_counts=True, axis=None)

        # Storing the # of unique classes in self.nc calculated from the classes returned by np.unique().
        self.nc = len(classes)

        if printFlag:
            print("---------------------------------------------------")
            print(f"  Shape of Input Data: {train_data_np.shape}")
            print(f"  Number of Data Points: {self.n_train}")
            print(f"  Number of Input Features: {self.d}")
            print(f"  Number of Target Classes: {self.nc}")
            print(f"  Shape of X_train: {X_train.shape}")
            print(f"  Shape of T_train: {T_train.shape}")
            print("---------------------------------------------------")

        return (self.n_train, X_train, T_train)

    def generateTestData(self, testData, printFlag):
        '''
        Transforms the test_data pandas dataframe into numpy array and splits it into features and true labels.

        Input -> test_data which is a pandas dataframe of testing data.
        Output -> Tuple of input features of test data (X_test), # of features in the input data (n_test), 
        True labels vector for test data (T_test)

        (X_test, n_test, T_test).

        '''

        # Converting the test dataframe to numpy array.
        test_data_np = testData.to_numpy()

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
            print(f"  Shape of X_test: {X_test.shape}")
            print(f"  Shape of T_test: {T_test.shape}")
            print("---------------------------------------------------")

        return (self.n_test, X_test, T_test)

    def modelTrain(self, X_train, T_train, kernel, C, gamma = "scale"):

        self.support_vectors = []
        SVMClassifier = SVC(kernel=kernel, C=C, gamma=gamma)
        self.model = SVMClassifier.fit(X_train, T_train)

        if kernel == "linear":
            self.w_vector = self.model.coef_

        else:
            self.w_vector = self.model.dual_coef_
        self.w_knot = self.model.intercept_

        # Getting the support vector.
        self.support_vectors = self.model.support_vectors_

    def calculateAccuracy(self, X, T, n):
        accuracy = self.model.score(X, T)*100

        return accuracy

    def plotSVMBoundaries(self, training, label_train, classifier, support_vectors=[], fsize=(6, 4), legend_on=True):
        # Plot the decision boundaries and data points for minimum distance to
        # class mean classifier
        #
        # training: traning data
        # label_train: class lables correspond to training data
        # classifier: sklearn classifier model, must have a predict() function
        #
        # Total number of classes
        nclass = max(np.unique(label_train))

        # Set the feature range for ploting
        max_x = np.ceil(max(training[:, 0])) + 0.01
        min_x = np.floor(min(training[:, 0])) - 0.01
        max_y = np.ceil(max(training[:, 1])) + 0.01
        min_y = np.floor(min(training[:, 1])) - 0.01

        xrange = (min_x, max_x)
        yrange = (min_y, max_y)

        # step size for how finely you want to visualize the decision boundary.
        inc = 0.05

        # generate grid coordinates. this will be the basis of the decision
        # boundary visualization.
        (x, y) = np.meshgrid(np.arange(
            xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

        # size of the (x, y) image, which will also be the size of the
        # decision boundary image that is used as the plot background.
        image_size = x.shape
        # make (x,y) pairs as a bunch of row vectors.
        xy = np.hstack((x.reshape(x.shape[0]*x.shape[1], 1, order='F'),
                    y.reshape(y.shape[0]*y.shape[1], 1, order='F')))

        # distance measure evaluations for each (x,y) pair.
        pred_label = classifier.predict(xy)

        # reshape the idx (which contains the class label) into an image.
        decisionmap = pred_label.reshape(image_size, order='F')

        # documemtation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
        symbols_ar = np.array(['rx', 'bo', 'ms', 'cd', 'gp',
                            'y*', 'kx', 'gP', 'r+', 'bh'])
        mean_symbol_ar = np.array(
            ['rd', 'bd', 'md', 'cd', 'gd', 'yd', 'kd', 'gd', 'rd', 'bd'])
        markerfacecolor_ar = np.array(
            ['r', 'b', 'm', 'c', 'g', 'y', 'k', 'g', 'r', 'b'])

        # turn on interactive mode
        plt.figure(figsize=fsize)
        # plt.ion()

        # show the image, give each coordinate a color according to its class label
        plt.imshow(decisionmap, extent=[
                xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

        if len(support_vectors) > 0:
            sv_x = support_vectors[:, 0]
            sv_y = support_vectors[:, 1]
            plt.scatter(sv_x, sv_y, s=100, c='green')
            # plot the class training data.
        plot_index = 0
        class_list = []
        class_list_name = []  # for legend
        mean_list = []  # for legend
        mean_lis_name = []  # for legend
        for cur_label in np.unique(label_train):
            # print(cur_label,plot_index,np.sum(label_train == cur_label))
            d1, = plt.plot(training[label_train == cur_label, 0],
                        training[label_train == cur_label, 1], symbols_ar[plot_index])

            if legend_on:
                class_list.append(d1)
                class_list_name.append('Class '+str(plot_index))
                l = plt.legend(class_list, class_list_name, loc=2)
                plt.gca().add_artist(l)

            plot_index = plot_index + 1

            # # plot support vectors

        plt.show()

        # unique_labels = np.unique(label_train)
        # # plot the class training data.
        # plt.plot(training[label_train == unique_labels[0], 0],training[label_train == unique_labels[0], 1], 'rx')
        # plt.plot(training[label_train == unique_labels[1], 0],training[label_train == unique_labels[1], 1], 'go')
        # if nclass == 3:
        #     plt.plot(training[label_train == unique_labels[2], 0],training[label_train == unique_labels[2], 1], 'b*')

        # # include legend for training data
        # if nclass == 3:
        #     l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
        # else:
        #     l = plt.legend(('Class 1', 'Class 2'), loc=2)
        # plt.gca().add_artist(l)

        # # plot support vectors
        # if len(support_vectors)>0:
        #     sv_x = support_vectors[:, 0]
        #     sv_y = support_vectors[:, 1]
        #     plt.scatter(sv_x, sv_y, s = 100, c = 'blue')

        # plt.show()