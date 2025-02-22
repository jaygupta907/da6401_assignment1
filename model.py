from layer import Perceptron_Layer,ACTIVATIONS
import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []
        self.activations = []

    
    def add(self,layer,activation=None):
        self.layers.append(layer)
        if activation:
            self.activations.append(ACTIVATIONS[activation])
        else:
            self.activations.append(None)
    
    def forward(self,x):
        A = x.T
        for i,layer in enumerate(self.layers):
            Z = layer.forward(A)
            if self.activations[i]:
                activation_func,_= self.activations[i]
                A = activation_func(Z)
            else:
                A = Z
        return A.T