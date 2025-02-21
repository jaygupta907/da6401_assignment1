import numpy as np
import os
import sys
import matplotlib.pyplot as plt



class Perceptron_Layer:
    def __init__(self, input_dim,output_dim,weight_init='random'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if weight_init == 'random':
            self.weights = np.random.randn(output_dim,input_dim)
            self.biases = np.random.randn(output_dim,1)
        elif weight_init == 'xavier_uniform':
            limit = np.sqrt(6 / (input_dim + output_dim))  
            self.weights = np.random.uniform(-limit, limit, size=(output_dim, input_dim))
        elif weight_init == 'xavier_normal':
            std = np.sqrt(2 / (input_dim + output_dim))
            self.weights = np.random.normal(0, std, size=(output_dim, input_dim))
        else:
            raise ValueError('Weight initialization not supported')