from layer import ACTIVATIONS
import numpy as np
from loss import cross_entropy_derivative,cross_entropy_loss
import wandb

class Sequential:
    '''
    A class to represent a sequential model.
    '''
    def __init__(self,args):
        '''
        Constructor for the Sequential class.
        '''
        self.layers = []
        self.activation_functions = []
        self.args = args

    
    def add(self,layer,activation=None):
        '''
        Adds a layer to the model.

        Args:
            layer (Perceptron_Layer): The layer to be added.
            activation (str): The activation function to be used.
        '''
        self.layers.append(layer)
        if activation:
            self.activation_functions.append(ACTIVATIONS[activation])
        else:
            self.activation_functions.append(None)
    
    def forward(self,X):
        '''
        Forward pass of the model.
        
        Args:
            X (numpy.ndarray): Input, shape (batch_size, input_dim).
        
        Returns:
            numpy.ndarray: Output, shape (batch_size, num_classes).
        '''
        self.activations =[]
        A = X.T
        for i,layer in enumerate(self.layers):
            Z = layer.forward(A)
            if self.activation_functions[i]:
                activation_func,_= self.activation_functions[i]
                A = activation_func(Z)
            else:
                A = Z
            self.activations.append(A)
        return A.T
    
    def backward(self,X,Y):
        '''
        Backward pass of the model.
        
        Args:
            X (numpy.ndarray): Input, shape (batch_size, input_dim).
            Y (numpy.ndarray): True labels, shape (batch_size, num_classes).

        '''
        grad_output = cross_entropy_derivative(self.activations[-1].T,Y)
        for i in reversed(range(len(self.layers))):
            if self.activation_functions[i]:
                _,activation_grad = self.activation_functions[i]
                grad_output = activation_grad(self.activations[i]) * grad_output
            grad_output = self.layers[i].backward(grad_output)


    def train(self,train_batches,test_batches,args):
        '''
        Trains the model.
        
        Args:
            train_batches (list): List of training batches.
            test_batches (list): List of test batches.
            args (argparse.ArgumentParser): Command line arguments.
        '''
        for epoch in range(args.epochs):
            loss = 0
            for _,(X_batch, Y_batch) in enumerate(train_batches):
                Y_pred = self.forward(X_batch)
                self.backward(X_batch,Y_batch)
                loss += cross_entropy_loss(Y_pred,Y_batch)
            if epoch % args.eval_freq == 0:
                print("\n <=======================Evaulating=======================>")
                evaluation_accuracy = self.evaluate(test_batches)
                wandb.log({"epoch": epoch,"evaluation_accuracy": evaluation_accuracy})
                print("The evaluation accuracy at epoch {} is {} \n".format(epoch,evaluation_accuracy))
            wandb.log({"epoch": epoch,"Training_Loss": loss})
            print('The cross entropy loss at epoch {} is {}'.format(epoch,loss))


    def evaluate(self,batches):
        '''
        Evaluates the model.
        
        Args:
            batches (list): List of batches.
        '''
        correct = 0
        total = 0
        for _,(X_batch, Y_batch) in enumerate(batches):
            predictions = self.forward(X_batch)
            correct += np.sum(np.argmax(predictions,axis=1) == np.argmax(Y_batch,axis=1))
            total += self.args.batch_size
        accuracy = correct / total
        return accuracy
        