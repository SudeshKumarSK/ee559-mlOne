import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras import Model
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

class MyModel(Model):
  def __init__(self, hidden_units, lambd_a, num_pixels):
    super(MyModel, self).__init__()
    self.input_layer = Flatten(input_shape=(28, 28), name="Input_Layer")
    self.d1 = Dense(hidden_units, activation='relu', input_shape=(num_pixels, ), kernel_regularizer=keras.regularizers.l2(lambd_a), bias_regularizer=keras.regularizers.l2(lambd_a), name="Hidden_Layer")
    self.output_layer = Dense(10, kernel_regularizer=keras.regularizers.l2(lambd_a), bias_regularizer=keras.regularizers.l2(lambd_a), name="Output_Layer")
    self.call(Input(shape=(28, 28)))

  def call(self, x):
    x = self.input_layer(x)
    x = self.d1(x)
    return self.output_layer(x)
  

class Classifier():

    def __init__(self):
        self.nc = 0
        self.d = 0
        self.n_train = 0
        self.n_test = 0
        self.n_validation = 0

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.validation_loss = tf.keras.metrics.Mean(name='validation_loss')
        self.validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


        self.validation_accuracy_iter = float("-inf")



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
        print(f"Shape of Flattened Valdation data:  {validation_images_flattened.shape}")
        print(f"Shape of Flattened Test data:  {test_images_flattened.shape}")
        print(f"We have {self.d} features in the input data.")
        print("---------------------------------------------------------------------------------------------")

        return (train_images_flattened, validation_images_flattened, test_images_flattened, self.d)

    
    @tf.function
    def train_step(self, images, labels, model):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=False)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)       

    

    @tf.function
    def validation_step(self, images, labels, model):

        predictions = model(images, training=False)
        valid_loss = self.loss_object(labels, predictions)

        self.validation_loss(valid_loss)
        self.validation_accuracy(labels, predictions)
        
    @tf.function
    def test_step(self, images, labels, model):

        predictions = model(images, training=False)
        test_loss = self.loss_object(labels, predictions)

        self.test_loss(test_loss)
        self.test_accuracy(labels, predictions)


    def setBatchSize(self, batch_size, X_train, T_train, X_validation, T_validation):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, T_train)).batch(batch_size)
        validation_ds = tf.data.Dataset.from_tensor_slices((X_validation, T_validation)).batch(batch_size)

        return (train_ds, validation_ds)
    

    def setOptimizerLearningRate(self, learning_rate):
        
        self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)


    def model_train(self, train_ds, validation_ds):
        model = MyModel(48, 0.001, 784)
        EPOCHS = 100
        startTime = time.time()
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.validation_loss.reset_states()
            self.validation_accuracy.reset_states()

            for train_images, train_labels in train_ds:
                self.train_step(train_images, train_labels, model)

            for validation_images, validation_labels in validation_ds:
                self.validation_step(validation_images, validation_labels, model)

            print(
            f'Epoch {epoch + 1}, '
            f'Training Loss: {self.train_loss.result()}, '
            f'Training Accuracy: {self.train_accuracy.result() * 100}, '
            f'Validation Loss: {self.validation_loss.result()}, '
            f'Validation Accuracy: {self.validation_accuracy.result() * 100}'
            )

            if self.validation_accuracy.result() > 0.8:
                stopTime = time.time()
                timeTaken = stopTime - startTime
                print(f"Time taken: {timeTaken}")
                print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                return timeTaken
            

    def training_loop(self, model, train_ds, validation_ds, hidden_units, reg_param, learning_rate):
        EPOCHS = 30
        train_acc = []
        val_acc = []
        train_loss = []
        val_loss = []
        startTime = time.time()

        self.setOptimizerLearningRate(learning_rate)
        for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.validation_loss.reset_states()
            self.validation_accuracy.reset_states()

            for train_images, train_labels in train_ds:
                self.train_step(train_images, train_labels, model)

            for validation_images, validation_labels in validation_ds:
                self.validation_step(validation_images, validation_labels, model)

            train_acc.append(self.train_accuracy.result() * 100)
            train_loss.append(self.train_loss.result())

            val_acc.append(self.validation_accuracy.result() * 100)
            val_loss.append(self.validation_loss.result())

        stopTime = time.time()
        timeTaken = stopTime - startTime
        print(
            f'Epoch -> {30}, '
            f'Validation Accuracy: {self.validation_accuracy.result() * 100}'
            )
        print(f"Time taken to finish 30 EPOCHS on hidden units - {hidden_units}, reg param = {reg_param} and learning rate - {learning_rate} ===> {timeTaken} secs")

        return (train_acc, train_loss, val_acc, val_loss, self.validation_accuracy.result() * 100)


    def training_loop_c(self, train_ds, validation_ds, hidden_units, lambd_a, learning_rate, EPOCHS, iteration):
        
        result = {}
        best_val_acc = float("-inf")

        model = MyModel(hidden_units, lambd_a, 784)
        self.setOptimizerLearningRate(learning_rate)

        startTime = time.time()
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        
        for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.validation_loss.reset_states()
            self.validation_accuracy.reset_states()

            for train_images, train_labels in train_ds:
                self.train_step(train_images, train_labels, model)

            for validation_images, validation_labels in validation_ds:
                self.validation_step(validation_images, validation_labels, model)
            
            print(
            f'Epoch {epoch + 1}, '
            f'Training Loss: {self.train_loss.result()}, '
            f'Training Accuracy: {self.train_accuracy.result() * 100}, '
            f'Validation Loss: {self.validation_loss.result()}, '
            f'Validation Accuracy: {self.validation_accuracy.result() * 100}'
            )
            

            if  self.validation_accuracy.result() > self.validation_accuracy_iter:
                self.validation_accuracy_iter = self.validation_accuracy.result()
                model.save_weights("./Weights/model_weights")
                print(f"Model saved for EPOCH: {epoch+1} at ITERATION: {iteration} with Validation Accuracy: {self.validation_accuracy_iter*100}%")
                
                

            if self.validation_accuracy.result() > best_val_acc:
                best_val_acc = self.validation_accuracy.result()
                result["epoch"] = epoch + 1
                result["best_validation_accuracy"] = best_val_acc*100
                print(f"Highest Accuracy so far: {self.validation_accuracy.result()*100}% at EPOCH: {epoch + 1} in ITERATION: {iteration}")

            print()

        stopTime = time.time()
        timeTaken = stopTime - startTime
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print()
        print(f"Time taken to finish {EPOCHS} epochs ===> {timeTaken} secs")
        print()

        return result


    def plotGraphs(self, hidden_units, learning_rate, reg_param, val):
        # Create the subplots

        fig, axs = plt.subplots(2, 2, figsize=(18, 10))

        # Iterate over the subplots and plot the data
        for i, ax in enumerate(axs.flat):
            if i == 0:
                ax.plot([ep for ep in range(1, 31)], val[0], label="Train Accuracy", color = "#66347F")
                ax.scatter([p for p in range(1, 31)], val[0], color = "#66347F", marker='o', s = 30, alpha=1)
                ax.set_ylabel('Train Accuracy(%)')
                ax.set_xlabel('Epochs')
                ax.set_title(f"Hidden Units: {hidden_units}, Learning rate: {learning_rate}, lambda: {reg_param}")
                ax.legend(["Accuracy"], loc = 0, frameon = True)
                
            elif i == 1:
                ax.plot([ep for ep in range(1, 31)], val[1], label="Train Loss", color = "#EA5455")
                ax.scatter([p for p in range(1, 31)], val[1], color = "#EA5455", marker='o', s = 30, alpha=1)
                ax.set_ylabel('Training Loss')
                ax.set_xlabel('Epochs')
                ax.set_title(f"Hidden Units: {hidden_units}, Learning rate: {learning_rate}, lambda: {reg_param}")
                ax.legend(["Loss"], loc = 0, frameon = True)

            elif i == 2:
                ax.plot([ep for ep in range(1, 31)], val[2], label="Validation Accuracy", color = "#52734D")
                ax.scatter([p for p in range(1, 31)], val[2], color = "#52734D", marker='o', s = 30, alpha=1)
                ax.set_ylabel('Validation Accuracy(%)')
                ax.set_xlabel('Epochs')
                ax.set_title(f"Hidden Units: {hidden_units}, Learning rate: {learning_rate}, lambda: {reg_param}")
                ax.legend(["Accuracy"], loc = 0, frameon = True)

            else:
                ax.plot([ep for ep in range(1, 31)], val[3], label="Validation Loss", color = "#F07B3F")
                ax.scatter([p for p in range(1, 31)], val[3], color = "#F07B3F", marker='o', s = 30, alpha=1)
                ax.set_ylabel('Validation Loss')
                ax.set_xlabel('Epochs')
                ax.set_title(f"Hidden Units: {hidden_units}, Learning rate: {learning_rate}, lambda: {reg_param}")
                ax.legend(["Loss"], loc = 0, frameon = True)


        # Show the plot
        plt.show()