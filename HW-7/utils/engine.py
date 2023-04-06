import tensorflow as tf
import numpy as np
import time


class Classifier():

    def __init__(self):
        self.nc = 0
        self.d = 0
        self.n_train = 0
        self.n_test = 0
        self.n_validation = 0


    def splitTrainData(self, train_images, train_labels, test_images, validation_split=0.2, printFlag = False): 
        num_validation_samples = int(validation_split * len(train_images))

        validation_images = train_images[:num_validation_samples]
        validation_labels = train_labels[:num_validation_samples]

        train_images = train_images[num_validation_samples:]
        train_labels = train_labels[num_validation_samples:]
        
        
        self.nc = len(np.unique(train_labels))
        self.n_train = train_images.shape[0]
        self.n_validation = validation_images.shape[0]
        self.n_test = test_images.shape[0]

        if printFlag:
            print("---------------------------------------------------------------------------------------------")
            print(f"Total number of images in Training data:  {self.n_train}")
            print(f"Total number of images in Validation data:  {self.n_validation}")
            print(f"Total number of images in Test data:  {self.n_test}")
            print(f"Total number of classes in the output lables: {self.nc}")
            print("---------------------------------------------------------------------------------------------")

        return ((train_images, train_labels), (validation_images, validation_labels))
        


    def flattenData(self, train_images, validation_images, test_images):
        train_images_flattened = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2])
        test_images_flattened = test_images.reshape(test_images.shape[0], train_images.shape[1]*train_images.shape[2])
        validation_images_flattened = validation_images.reshape(validation_images.shape[0], train_images.shape[1]*train_images.shape[2])

        self.d = train_images_flattened.shape[1]

        print("---------------------------------------------------------------------------------------------")
        print(f"Shape of Flattened Training data:  {train_images_flattened.shape}")
        print(f"We have {self.d} features in the input data.")
        print("---------------------------------------------------------------------------------------------")

        return (train_images_flattened, validation_images_flattened, test_images_flattened, self.d)