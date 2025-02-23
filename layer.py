import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import wandb

ACTIVATIONS = {
    "relu": (
        lambda x: np.maximum(0, x),
        lambda x: (x > 0).astype(float)
    ),
    "sigmoid": (
        lambda x: 1 / (1 + np.exp(-x)),
        lambda x: lambda y: y * (1 - y)  
    ),
    "tanh": (
        lambda x: np.tanh(x),
        lambda x: lambda y: 1 - y**2  
    ),
    "softmax": (
        lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True),
        lambda x: x  # Softmax derivative handled separately in loss
    )
}


class Perceptron_Layer:
    '''
    A class to represent a perceptron layer.
    '''
    def __init__(self, input_dim,output_dim,weight_init='random'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if weight_init == 'random':
            self.weights = np.random.randn(output_dim,input_dim)
        elif weight_init == 'xavier_uniform':
            limit = np.sqrt(6 / (input_dim + output_dim))  
            self.weights = np.random.uniform(-limit, limit, size=(output_dim, input_dim))
        elif weight_init == 'xavier_normal':
            std = np.sqrt(2 / (input_dim + output_dim))
            self.weights = np.random.normal(0, std, size=(output_dim, input_dim))
        else:
            raise ValueError('Weight initialization not supported')
        
        self.biases =  np.random.randn(output_dim,1)
        self.input = None
        self.timestep =0

        self.v_w =  np.zeros_like(self.weights)
        self.v_b =  np.zeros_like(self.biases)
        self.m_w =  np.zeros_like(self.weights)
        self.m_b =  np.zeros_like(self.biases)


    def forward(self, x):
        '''
        Forward pass of the perceptron layer.

        Args:
            x (numpy.ndarray): Input, shape (input_dim, batch_size).

        Returns:
            numpy.ndarray: Output, shape (output_dim, batch_size).
        '''
        self.input = x  #shape : (input_dim,batch_size)
        return np.dot(self.weights, x) + self.biases
    
    def backward(self, grad_output,args):
        """
        Computes and updates weights using the selected optimizer.

        Args:
            grad_output (numpy.ndarray): Gradient of loss w.r.t output, shape (out_dim, batch_size).

        Returns:
            numpy.ndarray: Gradient of loss w.r.t input, shape (in_dim, batch_size).
        """
        self.timestep += 1
        batch_size = args.batch_size
        grad_input = np.dot(self.weights.T, grad_output)  #shape : (input_dim,batch_size)
        grad_weights = np.dot(grad_output, self.input.T) / batch_size  #shape : (output_dim,input_dim)
        grad_biases = np.sum(grad_output, axis=1, keepdims=True) / batch_size  #shape : (output_dim,1)
        grad_weights += args.weight_decay * self.weights
        grad_biases  += args.weight_decay * self.biases 

        if args.optimizer == 'sgd':
            self.weights -= args.learning_rate * grad_weights
            self.biases  -= args.learning_rate * grad_biases
            
        if args.optimizer == 'momentum':
            self.v_w = args.momentum * self.v_w +grad_weights
            self.v_b = args.momentum * self.v_b +grad_biases
            self.weights -= args.learning_rate*self.v_w
            self.biases  -= args.learning_rate*self.v_b

        if args.optimizer == 'nag':
            pass
        if args.optimizer == 'adam':
            pass
        if args.optimizer == 'rmsprop':
            pass
        if args.optimizer =='nadam':
            pass

        return grad_input


        


