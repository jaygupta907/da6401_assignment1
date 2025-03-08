from layer import ACTIVATIONS 
import numpy as np  
from loss import cross_entropy_derivative, cross_entropy_loss ,loss_derivative_wrt_activation ,mse_derivative,mean_squared_loss
import wandb  
from sklearn.metrics import confusion_matrix  
import plotly.express as px  
import pandas as pd 
import plotly.graph_objects as go
import plotly.io as pio  

class Sequential:
    """
    A class to represent a sequential neural network model.
    """
    def __init__(self, args):
        """
        Initializes the sequential model.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        self.layers = []  # List to store layers
        self.activation_functions = []  # List to store activation functions
        self.args = args  # Store command-line arguments
        if self.args.loss == 'cross_entropy':
            self.loss = cross_entropy_loss
            self.loss_derivative = cross_entropy_derivative
        elif self.args.loss == 'mse':
            self.loss = mean_squared_loss
            self.loss_derivative = mse_derivative
        

    def add(self, layer, activation=None):
        """
        Adds a layer to the model.
        
        Args:
            layer (Perceptron_Layer): The layer to be added.
            activation (str, optional): The activation function to be used.
        """
        self.layers.append(layer)  # Append layer to the list
        if activation:
            self.activation_functions.append(ACTIVATIONS[activation])  # Store activation function
        else:
            self.activation_functions.append(None)  # No activation function

    def forward(self, X):
        """
        Performs the forward pass of the model.
        
        Args:
            X (numpy.ndarray): Input, shape (batch_size, input_dim).
        
        Returns:
            numpy.ndarray: Output, shape (batch_size, num_classes).
        """
        self.activations = []  # Store intermediate activations
        A = X.T  # Transpose input for matrix operations
        for i, layer in enumerate(self.layers):
            Z = layer.forward(A)  # Forward pass through layer
            if self.activation_functions[i]:
                activation_func, _ = self.activation_functions[i]
                A = activation_func(Z)  # Apply activation function
            else:
                A = Z  # No activation function
            self.activations.append(A)  # Store activation output
        return A.T  # Transpose output back

    def backward(self, X, Y):
        """
        Performs the backward pass of the model.
        
        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.
        """
        grad_output = self.loss_derivative(self.activations[-1].T, Y)
        grad_output = loss_derivative_wrt_activation(self.activations[-1].T,grad_output)
        for i in reversed(range(len(self.layers))):  # Iterate through layers in reverse order
            if self.activation_functions[i]:
                _, activation_grad = self.activation_functions[i]
                grad_output = activation_grad(self.activations[i]) * grad_output  # Apply activation derivative
            grad_output = self.layers[i].backward(grad_output)  # Backpropagate through layer

    def train(self, train_batches, test_batches, val_batches):
        """
        Trains the model.
        
        Args:
            train_batches (list): List of training batches.
            test_batches (list): List of test batches.
            val_batches (list): List of validation batches.
        """
        for epoch in range(self.args.epochs):
            loss = 0
            correct = 0
            total = 0
            for _, (X_batch, Y_batch) in enumerate(train_batches):
                total += self.args.batch_size
                Y_pred = self.forward(X_batch)  # Perform forward pass
                self.backward(X_batch, Y_batch)  # Perform backward pass
                loss += self.loss(Y_pred, Y_batch)  # Compute loss
                correct += np.sum(np.argmax(Y_pred, axis=1) == np.argmax(Y_batch, axis=1))  # Compute accuracy
            
            if epoch % self.args.eval_freq == 0:
                evaluation_accuracy, evaluation_loss = self.evaluate(val_batches, 'validation')
                wandb.log({"epoch": epoch, "evaluation_accuracy": evaluation_accuracy})
                wandb.log({"epoch": epoch, "evaluation_loss": evaluation_loss})
            
            loss = loss / len(train_batches)  # Compute average loss
            accuracy = correct / total  # Compute accuracy

            wandb.log({"epoch": epoch, "Training_Loss": loss})
            wandb.log({"epoch": epoch, "Training_Accuracy": accuracy})
            
            print(f'The training cross entropy loss at epoch {epoch} is {loss}')
            print(f'The training accuracy at epoch {epoch} is {accuracy}')

    def evaluate(self, batches, batch_type, classes=None):
        """
        Evaluates the model on given batches.
        
        Args:
            batches (list): List of batches.
            batch_type (str): Type of dataset (train, validation, or test).
            classes (list, optional): Class labels.

        Returns:
            tuple: Accuracy and loss.
        """
        correct = 0
        total = 0
        loss = 0
        y_true =[]
        y_pred =[]
        for _,(X_batch, Y_batch) in enumerate(batches):
            predictions = self.forward(X_batch)
            loss += self.loss(predictions,Y_batch)
            correct += np.sum(np.argmax(predictions,axis=1) == np.argmax(Y_batch,axis=1))
            y_true.append(np.argmax(Y_batch,axis=1))
            y_pred.append(np.argmax(predictions,axis=1))
            total += self.args.batch_size
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        loss = loss/len(batches)
        accuracy = correct / total
        if (batch_type=='test'):
            cm = confusion_matrix(y_true, y_pred)

            fig = go.Figure()
            hovertext = [[f"True: {classes[i]}<br>Predicted: {classes[j]}<br>Count: {cm[i, j]}"
              for j in range(len(classes))] for i in range(len(classes))]
            fig.add_trace(go.Heatmap(
                z=cm, 
                x=classes, 
                y=classes,
                colorscale="Blues",  
                text=cm,  
                texttemplate="%{text}", 
                hoverinfo="text",
                hovertext = hovertext 
            ))

            fig.update_layout(
                title="Confusion Matrix",
                xaxis=dict(title="Predicted Label", tickangle=-45, tickfont=dict(size=12)),
                yaxis=dict(title="True Label", tickfont=dict(size=12)),
                font=dict(size=14),
                width = 800,
                height = 800
            )
            wandb.log({"Confusion Matrix": wandb.Html(pio.to_html(fig,full_html=False))})


        return accuracy ,loss
        