# Importing all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



class OneVsRest():

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
    

    def analyseData(self, T_train, T_test):
        pass

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
    

    def changeLabels(self, T, label):

        T_changed = np.copy(T)

        T_changed[T == label] = 1.0
        T_changed[T != label] = -1.0

        return T_changed
    

    def computeCost(self, X_train, T_train, n_train, w_vector):
        J = 0

        for n in range(n_train):
            curr_loss = np.dot(w_vector, X_train[n]) * T_train[n]

            if curr_loss < 0:
                J = J + (curr_loss)

        return -1*J

    
    def modelTrain_SGD(self, X_train, T_train, n_train, w_vector, labels, epochs = 100, learn_rate = 1, printFlag = True):

        final_w_vector = np.ones((self.d + 1, self.nc))
        T_train_labels = np.zeros((n_train, self.nc))
        T_train_labels_shuffled = np.zeros((n_train, self.nc))
        
        
        for i, label in enumerate(labels):
            T_train_labels[:, i] = self.changeLabels(T=T_train, label=i)
            epoch = 0
            cost = float("inf")
            while epoch < epochs:
                X_train_shuffled, T_train_labels_shuffled[:, i] = self.shuffleData(X=X_train, T=T_train_labels[:, i])
                for n in range(n_train):

                    curr_loss = np.dot(w_vector[:, i], X_train_shuffled[n]) * T_train_labels_shuffled[:, i][n]

                    if curr_loss < 0:
                        w_vector[:, i] = w_vector[:, i] - learn_rate * (-1 * X_train_shuffled[n] * T_train_labels_shuffled[:, i][n])


                    else:
                        w_vector[:, i] = w_vector[:, i]


                    if epoch == 99 and n >= self.n_train - 100:
                        current_cost = self.computeCost(w_vector = w_vector[:, i], X_train = X_train, T_train = T_train_labels[:, i], n_train=n_train)
                        
                        if current_cost < cost:
                            cost = current_cost

                            if printFlag:
                                print(f"Epoch #{epoch+1} and Interation #{n}of Classifier - {i+1} --> Cost J(W) is: {current_cost}")

                            final_w_vector[:, i] = w_vector[:, i]

                if printFlag:
                    print(f"Epoch #{epoch+1} of Classifier - {i+1} --> Cost J(W) is: {self.computeCost(w_vector = w_vector[:, i], X_train = X_train, T_train = T_train_labels[:, i], n_train=n_train)}")
                epoch += 1

            print(f"Optimum w_vector for the {i+1}th classifier is: {final_w_vector[:, i]}")
    
        return final_w_vector


    def predict(self, optimum_w_vector, X_train):

        Y_hat = np.dot(optimum_w_vector, X_train.T)

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
    


    def predictionTechnique1(self, X, T, w_vector, n):
        totalPredictions = n
        correctClassifiedData = 0
        misClassifiedData = 0
        unClassifiedData = 0
        #  7 x 17 . (17 x 1)
        for i in range(n):
            g_x = w_vector.T @ X[i]
            preds = np.sign(g_x)

            if np.sum(preds==1) == 1:
                if np.squeeze(np.where(preds == 1)[0]) == T[i]:
                    correctClassifiedData += 1

                else:
                    misClassifiedData += 1

            else:
                unClassifiedData += 1

        return (((correctClassifiedData / totalPredictions) * 100), ((misClassifiedData / totalPredictions) * 100), ((unClassifiedData / totalPredictions) * 100))

                
    def predictionTechnique2(self, X, T, w_vector, n):
        totalPredictions = n
        correctClassifiedData = 0
        misClassifiedData = 0
        unClassifiedData = 0
        
        # Matrix Multiplication -> (7, 17) x (17, n_train)
        g_x = (w_vector.T @ X.T).T

        preds = np.argmax(g_x, axis = 1)

        correctClassifiedData = np.count_nonzero(np.equal(preds, T))

        misClassifiedData = n - correctClassifiedData

        unClassifiedData = n - (correctClassifiedData + misClassifiedData)

        return (((correctClassifiedData / totalPredictions) * 100), ((misClassifiedData / totalPredictions) * 100), ((unClassifiedData / totalPredictions) * 100))



    def predictionTechnique3(self, X, T, w_vector, n):
        totalPredictions = n
        correctClassifiedData = 0
        misClassifiedData = 0
        unClassifiedData = 0
        
        # Matrix Multiplication -> (7, 17) x (17, n_train) => 7 X n_train So basically 7 rows and 12000 columns.
        # I have taken a Transpose to get the dimension as (12000 x 7)
        g_x = (w_vector.T @ X.T).T

        # We have g_x which is (12000 x 7). We must divide g_x by the l2norm of w_vector where w_vector is (17 x 7)
        # If we use non-augmented weights, the dimension will be (16 x 7). WHen we take transpose of the w_vector non-augmented,
        # we get (7 X 16). After taking L2 Norm across the rows, we get a matrix of shape (7,). I reshaped it into (1,7)
        # Now we can divide g_x (12000 x 7) by w_l2norm (1 x 7). Broadcasting will take place .

        w_vector_L2Norm = np.linalg.norm(w_vector.T[:, 1:], axis = 1).reshape(1, 7)
        g_x_DividedByL2Norm = g_x / w_vector_L2Norm

        preds = np.argmax(g_x_DividedByL2Norm, axis = 1)

        correctClassifiedData = np.count_nonzero(np.equal(preds, T))

        misClassifiedData = n - correctClassifiedData

        unClassifiedData = n - (correctClassifiedData + misClassifiedData)

        return (((correctClassifiedData / totalPredictions) * 100), ((misClassifiedData / totalPredictions) * 100), ((unClassifiedData / totalPredictions) * 100))




    def plot_multiclass_histograms(self, X_aug, W, y, fname, norm_W=False, scale=1, class_names=None):
        """
        Keith Chugg, USC, 2023.

        X_aug: shape: (N, D + 1).  Augmented data matrix
        W: shape: (D + 1, C).  The matrix of augmented weight-vectors.  W.T[m] is the weight vector for class m
        y: length N array with int values with correct classes.  Classes are indexed from 0 up.
        fname: a pdf of the histgrams will be saved to filename fname
        norm_W: boolean.  If True, the w-vectors for each class are normalized.
        scale: use scale < 1 to make the figure smaller, >1 to make it bigger
        class_names: pass a list of text, descriptive names for the classes.  

        This function takes in the weight vectors for a linear classifier and applied the "maximum value methd" -- i.e., 
        it computes the argmax_m g_m(x), where g_m(x) = w_m^T x to find the decision. For each class, it plots the historgrams 
        of  g_m(x) when class c is true.  This gives insights into which classes are most easily confused -- i.e., similar to a 
        confusion matrix, but more information.  

        Returns: the overall misclassification error percentage
        """
        if norm_W:
            W = W / np.linalg.norm(W, axis=0)
        y_soft = X_aug @ W
        N, C = y_soft.shape
        y_hard = np.argmax(y_soft, axis=1)
        error_percent = 100 * np.sum(y != y_hard) / len(y) 

        fig, ax = plt.subplots(C, sharex=True, figsize=(12 * scale, 4 * C * scale))
        y_soft_cs = []
        conditional_error_rate = np.zeros(C)
        if class_names is None:
            class_names = [f'Class {i}' for i in range(C)]
        for c_true in range(C):
            y_soft_cs.append(X_aug[y == c_true] @ W)
            y_hard_c = np.argmax(y_soft_cs[c_true], axis=1)
            conditional_error_rate[c_true] = 100 * np.sum(y_hard_c != c_true) / len(y_hard_c)
        for c_true in range(C):
            peak = -100
            for c in range(C):
                hc = ax[c_true].hist(y_soft_cs[c_true].T[c], bins = 100, alpha=0.4, label=class_names[c])
                peak = np.maximum(np.max(hc[0]), peak)
                ax[c_true].legend()
                ax[c_true].grid(':')
            ax[c_true].text(0, 0.9 * peak, f'True: {class_names[c_true]}\nConditional Error Rate = {conditional_error_rate[c_true] : 0.2f}%')
        if norm_W:
            ax[C-1].set_xlabel(r'nromalized discriminant function $g_m(x) / || {\bf w} ||$')
        else:
            ax[C-1].set_xlabel(r'discriminant function $g_m(x)$')
        plt.savefig(fname, bbox_inches='tight',)
        return error_percent
