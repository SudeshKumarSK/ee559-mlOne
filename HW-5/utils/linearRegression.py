################################################
## EE559 HW-5.
## Created by Sudesh Kumar Santhosh Kumar.
## Tested in Python 3.10.9 using conda environment version 22.9.0.
################################################

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge


class Regressor():

    def __init__(self):
        self.nc = 0
        self.n_train = 0
        self.n_test = 0
        self.d = 0
        self.d_poly = [0, 0, 0, 0, 0, 0, 0, 0]


    def generateTrainData(self, trainData, printFlag = False):
        
        self.d = trainData.shape[1] - 1

        self.n_train = trainData.shape[0]

        # Converting the sorted dataframe to numpy array.
        train_data_np = trainData.to_numpy()


        X_train = train_data_np[:, 0:-1]
        T_train = train_data_np[:, -1:]

        # Finding the number of unique labels in the Target (T)
        classes = np.unique(T_train, axis=None)

        self.nc = len(classes)

        if printFlag:
            print("------------------------------------------------------------------------------------------")
            print(f"TRAIN DATA ANALYSIS: ")
            print(f"  Shape of Input Data: {train_data_np.shape}")
            print(f"  Number of Data Points: {self.n_train}")
            print(f"  Number of Input Features: {self.d}")
            print(f"  Number of Target Classes: {self.nc}")
            print("------------------------------------------------------------------------------------------")


        return (self.n_train, X_train, T_train)



    def generateTestData(self, testData, printFlag = False):

        self.n_test = testData.shape[0]

        test_data_np = testData.to_numpy()

        # Spliting the data_np into input features (X) and output target (T) basically splitting labels and features.
        X_test = test_data_np[:, 0:self.d]
        T_test = test_data_np[:, -1:]


        if printFlag:
            print("------------------------------------------------------------------------------------------")
            print(f"TEST DATA ANALYSIS: ")
            print(f"  Shape of Input Data: {test_data_np.shape}")
            print(f"  Number of Data Points: {self.n_test}")
            print(f"  Number of Input Features: {self.d}")
            print(f"  Number of Target Classes: {self.nc}")
            print("------------------------------------------------------------------------------------------")


        return (self.n_test, X_test, T_test)
    

    '''def generatePolynomialFeatures(self, X_train, degree, printFlag = False):

        poly = PolynomialFeatures(degree = degree, include_bias = True)

        X_train_poly = poly.fit_transform(X_train)

        if self.d_poly[degree] == 0:
            self.d_poly[degree] = X_train_poly.shape[1]

        if printFlag:
            print("------------------------------------------------------------------------------------------")
            print(f"Train DATA ANALYSIS AFTER POLYNOMIAL TRANSFORMATION of degree -> {degree}: ")
            print(f"  Shape of Train Data: {X_train_poly.shape}")
            print(f"  Number of Data Points: {self.n_train}")
            print(f"  Number of Input Features: {self.d_poly[degree]}")
            print(f"  Number of Target Classes: {self.nc}")
            print("------------------------------------------------------------------------------------------")

        return X_train_poly'''
    

    def generatePolynomialFeatures(self, X, n, degree, datasetName, printFlag = False):

        poly = PolynomialFeatures(degree = degree, include_bias = True)

        X_poly = poly.fit_transform(X)


        self.d_poly[degree] = X_poly.shape[1]

        if printFlag:
            print("------------------------------------------------------------------------------------------")
            print(f"{datasetName} ANALYSIS AFTER POLYNOMIAL TRANSFORMATION of degree -> {degree}: ")
            print(f"  Shape of Data: {X_poly.shape}")
            print(f"  Number of Data Points: {n}")
            print(f"  Number of Input Features: {self.d_poly[degree]}")
            print(f"  Number of Target Classes: {self.nc}")
            print("------------------------------------------------------------------------------------------")

        return X_poly
    

    def changeLabels(self, T, n):

        T_changed = np.zeros((n, 1))

        T_changed[T == 0.0] = 1.0
        T_changed[T == 1.0] = -1.0

        return T_changed
    

    def modelTrain(self, X_train, T_train):

        LR = LinearRegression(fit_intercept = False)

        model = LR.fit(X_train, T_train)

        w_vector = model.coef_

        return (model, w_vector)
    

    def modelTrain_Regularization(self, X_train, T_train, lambd_a):

        LR_Regularized = Ridge(alpha = lambd_a, fit_intercept = False)

        model_R = LR_Regularized.fit(X_train, T_train)
        # LR_Regularized.fit(X_train, T_train)

        w_vector =  model_R.coef_

        print(f"Training Process with Regularization, lambda = {lambd_a} Completed Successfully!")
        return (model_R, w_vector)
    

    def predictTrain(self, X, T, n, p, model, datasetName, printFlag = False):

        '''
        Calculates the Classifier's Accuracy from the Target Vector(T) and Prediction Vector(Y_hat)

        Input -> Target Vector(T), Prediction Vector(Y_hat) and a percentageFlag to display CER in percentage or not.
        Output -> Y_hat, Accuracy in percentage.
        
        '''

        predictions = model.predict(X)

        Y_hat = np.zeros((predictions.shape[0], 1))

        Y_hat[predictions >= 0.0] = 1.0
        Y_hat[predictions < 0.0] = -1.0


        totalPredictions = n
        correctPredictions = 0

        for i in range(len(Y_hat)):
            if T[i] == Y_hat[i]:
                correctPredictions += 1 

        acc = (correctPredictions / totalPredictions) * 100

        if printFlag:
            print("------------------------------------------------------------------------------------------")
            print(f"p = {p}")
            print(f"Accuracy on the {datasetName} for p -> {p} is: {acc}%")
            print("------------------------------------------------------------------------------------------")


        return (predictions, acc)
    
    
    def computeCost(self, T_train, Y_hat):
        mse = np.square(np.subtract(T_train, Y_hat)) 

        cost = np.mean(mse)

        return cost
    

    def computeCostRegularized(self, T_train, Y_hat, lambd_a, model):
        mse = np.square(np.subtract(T_train, Y_hat)) + (lambd_a*np.linalg.norm(model.coef_)**2)

        cost = np.mean(mse)

        return cost
    

    def printAccuracy(self, accuracyTrain, accuracyTest, p):
        print("----------------------------------------------------------------------------------------------------------------")
        print(f"Accuracy on the Training Set for p - {p}: ")
        print(accuracyTrain)
        print(f"Accuracy on the Test Set for p - {p}: ")
        print(accuracyTest)
        print("----------------------------------------------------------------------------------------------------------------")


    

    def plotAccuracy(self, train_Accuracy_History, test_Accuracy_History):
        ax = plt.axes()
        ax.plot([p for p in range(1, 8)], train_Accuracy_History, color = "#804674")
        ax.plot([p for p in range(1, 8)], test_Accuracy_History, color = "#609966")

        ax.scatter([p for p in range(1, 8)], train_Accuracy_History, color = "#804674", marker='o', s = 30, alpha=1)
        ax.scatter([p for p in range(1, 8)], test_Accuracy_History, color = "#609966", marker='o', s = 30, alpha=1)

        ax.set_title("Train Accuracy and Test Accuracy Vs. degree (p)")
        ax.set_ylabel('Accuracy (in Percentage %)')
        ax.set_xlabel('Degree (p)')
        ax.legend(["Train Accuracy", "Test Accuracy"], loc = 0, frameon = True)
        plt.show()



    def plotAccuracyRegularized(self, lambda_list, train_Accuracy_All, test_Accuracy_All):
        

        # Create the subplots
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))


        # Iterate over the subplots and plot the data 
        for i, ax in enumerate(axs.flat):
            ax.plot([p for p in range(1, 8)], train_Accuracy_All[i], label="Train Accuracy", color = "#804674")
            ax.plot([p for p in range(1, 8)], test_Accuracy_All[i], label="Test Accuracy", color = "#609966")

            ax.scatter([p for p in range(1, 8)], train_Accuracy_All[i], color = "#804674", marker='o', s = 30, alpha=1)
            ax.scatter([p for p in range(1, 8)], test_Accuracy_All[i], color = "#609966", marker='o', s = 30, alpha=1)

            ax.set_ylabel('Accuracy (in Percentage %)')
            ax.set_xlabel('Degree (p)')
            ax.set_title(f"Accuracy Vs. p for lambda = {lambda_list[i]}")
            ax.legend()

        # Show the plot
        plt.show()
        

    def plotCost(self, J_History):

        ax = plt.axes()

        ax.plot([p for p in range(1, 8)], J_History, color = "#DC3535")
    
        ax.scatter([p for p in range(1, 8)], J_History, color = "#DC3535", marker='o', s = 30, alpha=1)

        ax.set_title("JMSE Vs. degree (p)")
        ax.set_ylabel('Cost - JMSE')
        ax.set_xlabel('Degree (p)')
        ax.legend(["Cost JMSE"], loc = 0, frameon = True)

        plt.show()


    def plotCostRegularized(self, lambda_list, J_All):
        # Create the subplots
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))


        # Iterate over the subplots and plot the data 
        for i, ax in enumerate(axs.flat):
            ax.plot([p for p in range(1, 8)], J_All[i], label="Train Accuracy", color = "#804674")
            

            ax.scatter([p for p in range(1, 8)], J_All[i], color = "#804674", marker='o', s = 30, alpha=1)

            ax.set_ylabel('Cost - JMSE')
            ax.set_xlabel('Degree (p)')
            ax.set_title(f"JMSE Vs. p for lambda = {lambda_list[i]}")
            ax.legend(["Cost JMSE"], loc = 0, frameon = True)

        # Show the plot
        plt.show()
        

    def plotTestAccuracyVsLambda(self, lambda_List, test_Accuracy_All):
        

        lambda_List = np.log10(lambda_List)

        lambda_List = np.insert(lambda_List, 0, -1)

        print(lambda_List)

        ax = plt.axes()
        ax.plot(lambda_List, test_Accuracy_All[0], color = "#00A8B5")
        ax.plot(lambda_List, test_Accuracy_All[1], color = "#774898")
        ax.plot(lambda_List, test_Accuracy_All[2], color = "#DE4383")
        ax.plot(lambda_List, test_Accuracy_All[3], color = "#F3AE4B")
        
        ax.scatter(lambda_List, test_Accuracy_All[0], color = "#00A8B5", marker='o', s = 30, alpha=1)
        ax.scatter(lambda_List,test_Accuracy_All[1], color = "#774898", marker='o', s = 30, alpha=1)
        ax.scatter(lambda_List, test_Accuracy_All[2], color = "#DE4383", marker='o', s = 30, alpha=1)
        ax.scatter(lambda_List, test_Accuracy_All[3], color = "#F3AE4B", marker='o', s = 30, alpha=1)        

        ax.set_title("Accuracy Vs. log(lambda)")
        ax.set_ylabel('Accuracy (in Percentage %)')
        ax.set_xlabel('log(lambda)')
        plt.show()


    def plotNonLinear(self, X_train, T_train, degree, model):

        poly = PolynomialFeatures(degree = degree, include_bias = True)

        non_linear_trans = lambda x : poly.fit_transform(x)
        predictor = lambda x : np.where(model.predict(x) >= 0.0, 0.0, 1.0)                                                                      

        self.plotDecBoundaries_Nonlinear(X_train, T_train, non_linear_trans, predictor, fsize=(10,8), legend_on=True)



    def plotDecBoundaries_Nonlinear(self, feature, labels, non_linear_trans, predictor, fsize=(6,4),legend_on = False):
    
        '''
        Plot the decision boundaries and data points for any binary classifiers
        
        feature: origianl2D feautre, N x 2 array:
            N: number of data points
            2: number of features 
        labels: class lables correspond to feature, N x 1 array: [0,0,1,1,0,0,...]
            N: number of data points
        legend_on: add the legend in the plot. potentially slower for datasets with large number of clases and data points
        ----------------------------
        You need to write the following two functions

        non_linear_trans: your custom non-linear transforation function.
            <feature_nonlinear> = non_linear_trans(<feature_original>), 
                Input: <feature_original>, Nx2 array, 
                Output: <feature_nonlinear>: Nx? array.
            if no nonlinear transformation performs, then, 
            let non_linear_trans = lambda x:x, which just output your original feature
        
        predictor: your custom predictor.
            <predictions> = predictor(<feature>)
                Input: <feature> Nx? array.
                Output: <predictions> binary labels, i.e., array ([0,1,0,0,1...])

        If you don't want write custom functions, you can modify this plot function based on your need,
        do non-linear transformation and class prediction inside this plot function.
        ----------------------------
        '''

        labels = labels.astype(int)

        # Set the feature range for ploting
        max_x = np.ceil(max(feature[:, 0])) + 1
        min_x = np.floor(min(feature[:, 0])) - 1
        max_y = np.ceil(max(feature[:, 1])) + 1
        min_y = np.floor(min(feature[:, 1])) - 1

        xrange = (min_x, max_x)
        yrange = (min_y, max_y)

        # step size for how finely you want to visualize the decision boundary.
        inc = 0.05

        # generate grid coordinates. this will be the basis of the decision
        # boundary visualization.
        (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

        # size of the (x, y) image, which will also be the size of the
        # decision boundary image that is used as the plot background.
        image_size = x.shape
        xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) # make (x,y) pairs as a bunch of row vectors.
        
        '''
        You should write the custom functions, non_linear_trans and predictor
        '''
        # apply non-linear transformation to all points in the map (not only data points)
        xy = non_linear_trans(xy)
        # predict the class of all points in the map 
        pred_label = predictor(xy)

        # reshape the idx (which contains the class label) into an image.
        decisionmap = pred_label.reshape(image_size, order='F')

        # documemtation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
        symbols_ar = np.array(['rx', 'bo', 'ms', 'cd','gp','y*','kx','gP','r+','bh'])
        #show the image, give each coordinate a color according to its class label
        plt.figure(figsize=fsize)

        plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower', aspect='auto')

        # plot the class data.
        plot_index = 0
        class_list = []
        class_list_name = [] #for legend
        for cur_label in np.unique(labels):
            # print(cur_label,plot_index,np.sum(label_train == cur_label))
            d1, = plt.plot(feature[labels == cur_label, 0],feature[labels == cur_label, 1], symbols_ar[plot_index])

            if legend_on:
                class_list.append(d1)
                class_list_name.append('Class '+str(plot_index))
                l = plt.legend(class_list,class_list_name, loc=2)
                plt.gca().add_artist(l)
        
            plot_index = plot_index + 1

        plt.show()


    
    
    


