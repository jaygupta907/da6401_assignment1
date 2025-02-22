from layer import Perceptron_Layer,ACTIVATIONS
import numpy as np
from loss import cross_entropy_derivative,cross_entropy_loss


class Sequential:
    def __init__(self):
        self.layers = []
        self.activation_functions = []

    
    def add(self,layer,activation=None):
        self.layers.append(layer)
        if activation:
            self.activation_functions.append(ACTIVATIONS[activation])
        else:
            self.activation_functions.append(None)
    
    def forward(self,X):
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
        
        grad_output = cross_entropy_derivative(self.activations[-1],Y)
        for i in reversed(range(len(self.layers))):
            if self.activations[i]:
                _,activation_grad = self.activations[i]
                grad_output = activation_grad(self.activations[i]) * grad_output
            grad_output = self.layers[i].backward(grad_output)


    def train(self,train_batches,test_batches,args):
        for epoch in range(args.epochs):
            loss = 0
            for (X_batch,Y_batch) in zip(train_batches):
                Y_pred = self.forward(X_batch)
                self.backward(X_batch,Y_batch)
                loss += cross_entropy_loss(Y_pred,Y_batch)
            if epoch % args.eval_freq == 0:
                print("<=======================Evaulating=======================>")
                evaluation_accuracy = self.evaluate(test_batches)
                print("The evaluation accuracy at epoch {} is {} \n".format(epoch,evaluation_accuracy))
            print('The cross entropy loss at epoch {} is {}'.format(epoch,loss/args.batch_size))


    def evaluate(self,batches):
        correct = 0
        total = 0
        for (X_batch,Y_batch) in zip(batches):
            predictions = self.forward(X_batch)
            correct += np.sum(np.argmax(predictions,axis=1) == np.argmax(Y_batch,axis=1))
            total += len(Y_batch)
        accuracy = correct / total
        return accuracy
        