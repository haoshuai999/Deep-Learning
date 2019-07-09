# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

import keras
from keras.utils  import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

class SignLanguage:
    def __init__(self):
        self.model = None
        
        self.data = {
            "train": None,
            "test" : None
        }
        self.create_model()
    
    def create_model(self):
        """
        Create a CNN model and save it to self.model
        """
        
        model = Sequential()
        model.add(Conv2D(64,kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(28,28,1)))
        model.add(Conv2D(64,kernel_size=(5, 5),
         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128,kernel_size=(5, 5),
         activation='relu'))
        model.add(Conv2D(128,kernel_size=(5, 5),
         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile('adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        
        self.model = model
    
    def prepare_data(self, images, labels):
        """
        Use this method to normalize the dataset and split it into train/test.
        Save your data in self.data["train"] and self.data["test"] as a tuple
        of (images, labels)
        
        :param images numpy array of size (num_examples, 28*28)
        :param labels numpy array of size (num_examples, )
        """
        
        num_examples = images.shape[0]
        # print(images)
        mean = np.mean(images, axis=1)
        mean = mean.reshape(num_examples,1)
        # print(mean)
        std = np.std(images, axis=1)
        std = std.reshape(num_examples,1)
        # print(std)
        images = (images - mean) / std
        
        images = images.reshape(num_examples,28,28,1)
        
        x_train, x_test = np.split(images, [int(num_examples * 0.8)], axis=0)
        y_train, y_test = np.split(labels, [int(num_examples * 0.8)], axis=0)
#         print(x_train.shape)
#         print(y_train.shape)
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
        self.data = {
            "train": (x_train, y_train),
            "test" : (x_test, y_test)
        }
        
    
    def train(self, batch_size:int=128, epochs:int=50, verbose:int=1):
        """
        Use model.fit() to train your model. Make sure to return the history for a neat visualization.
        
        :param batch_size The batch size to use for training
        :param epochs     Number of epochs to use for training
        :param verbose    Whether or not to print training output
        """
        
        history = self.model.fit(self.data["train"][0],self.data["train"][1],
          batch_size=batch_size,
          epochs=epochs,
          verbose=verbose,
          validation_data=(self.data["test"][0],self.data["test"][1]))
        return history
    
    def predict(self, data):
        """
        Use the trained model to predict labels for test data.
        
        :param data: numpy array of test images
        :return a numpy array of test labels. array size = (num_examples, )
        """
        num_examples = data.shape[0]
        # print(data)
        mean = np.mean(data, axis=1)
        mean = mean.reshape(num_examples,1)
        # print(mean)
        std = np.std(data, axis=1)
        std = std.reshape(num_examples,1)
        # print(std)
        data = (data - mean) / std
        data = data.reshape(num_examples,28,28,1)
        # print(data)
#         print(data.shape[0])
        data = np.argmax( self.model.predict(data), axis=1)
        data = data.reshape(num_examples, )
        
#         print(np.zeros(data.shape[0]))

        return data
    
    def visualize_data(self, data):
        """
        Visualizing the hand gestures
        
        :param data: numpy array of images
        """
        if data is None: return
        
        nrows, ncols = 5, 5
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)

        for i in range(nrows):
            for j in range(ncols):
                axs[i][j].imshow(data[0][i*ncols+j].reshape(28, 28), cmap='gray')
        plt.show()

    def visualize_accuracy(self, history):
        """
        Plots out the accuracy measures given a keras history object
        
        :param history: return value from model.fit()
        """
        if history is None: return
        
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title("Accuracy")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train','test'])
        plt.show()


if __name__=="__main__":
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')

    train_labels, test_labels = train['label'].values, test['label'].values
    train.drop('label', axis=1, inplace=True)
    test.drop('label', axis=1, inplace=True)

    num_classes = test_labels.max() + 1
    train_images, test_images = train.values, test.values

    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

if __name__=="__main__":
    my_model = SignLanguage()
    my_model.prepare_data(train_images, train_labels)

if __name__=="__main__":
    my_model.visualize_data(my_model.data["train"])

if __name__=="__main__":
    history = my_model.train(epochs=30, verbose=1)
    my_model.visualize_accuracy(history)

if __name__=="__main__":
    y_pred = my_model.predict(test_images)
    accuracy = accuracy_score(test_labels, y_pred)
    print(accuracy)
