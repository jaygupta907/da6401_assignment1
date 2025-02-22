from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.datasets import fashion_mnist
import numpy as np
import os
import sys  



class Batch_Dataset:
    '''
    A class to represent a batch dataset.
    '''
    def __init__(self, dataset):
        if dataset == 'mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        elif dataset == 'fashion_mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
        else:
            raise ValueError('Dataset not supported')
        
    def preprocess(self):
        '''
        Preprocesses the dataset.
        '''
        self.x_train = self.x_train.reshape(-1,self.x_train.shape[1]*self.x_train.shape[2])
        self.x_test  = self.x_test.reshape(-1,self.x_test.shape[1]*self.x_test.shape[2])
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test  = self.x_test.astype('float32')  / 255.
        self.num_classes = len(np.unique(self.y_train))
        self.y_train = np.eye(self.num_classes)[self.y_train]
        self.y_test  = np.eye(self.num_classes)[self.y_test]



    def create_train_batches(self,batch_size=32, shuffle=True):
        '''
        Creates training batches.
        
        Args:
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the data.
        
        Returns:
            list: List of training batches.
        '''
        num_datapoints = len(self.x_train)
        indices = np.arange(num_datapoints)
        train_batches = []
        if shuffle:
            np.random.shuffle(indices) 
        for start_idx in range(0,num_datapoints, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            train_batches.append([self.x_train[batch_indices], self.y_train[batch_indices]])
        return train_batches

    def create_test_batches(self,batch_size=32, shuffle=True):
        '''
        Creates test batches.
        
        Args:
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            list: List of test batches
        '''
        num_datapoints = len(self.x_test)
        indices = np.arange(num_datapoints)
        test_batches = []
        if shuffle:
            np.random.shuffle(indices) 
        for start_idx in range(0,num_datapoints, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            test_batches.append([self.x_test[batch_indices], self.y_test[batch_indices]])
        return test_batches
