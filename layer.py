import numpy as np 
from optimizers import SGD, Momentum, Nesterov, Adagrad, RMSProp, Adam, Nadam  

# Define activation functions and their derivatives
ACTIVATIONS = {
    "relu": (
        lambda x: np.maximum(0, x),  # ReLU activation
        lambda x: (x > 0).astype(float)  # ReLU derivative
    ),
    "sigmoid": (
        lambda x: 1 / (1 + np.exp(-x)),  # Sigmoid activation
        lambda x: lambda y: y * (1 - y)  # Sigmoid derivative using output y
    ),
    "tanh": (
        lambda x: np.tanh(x),  # Tanh activation
        lambda x: lambda y: 1 - y**2  # Tanh derivative using output y
    ),
    "softmax": (
        lambda x: np.exp(x - np.max(x, axis=0, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=0, keepdims=True)), axis=0, keepdims=True),  # Softmax activation
        lambda x: np.ones_like(x)  # Placeholder derivative for softmax (handled in loss function)
    )
}

class Perceptron_Layer:
    """
    A class representing a fully connected perceptron layer in a neural network.
    """
    def __init__(self, input_dim, output_dim, args, weight_init='random'):
        """
        Initializes the perceptron layer with given parameters.
        
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output neurons.
            args (argparse.Namespace): Parsed arguments including optimizer and weight decay.
            weight_init (str): Method for weight initialization ('random', 'xavier_uniform', or 'xavier_normal').
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize weights based on the chosen method
        if weight_init == 'random':
            self.weights = np.random.normal(0,1,size=(output_dim,input_dim))
        elif weight_init == 'xavier_uniform':
            limit = np.sqrt(6 / (input_dim + output_dim))
            self.weights = np.random.uniform(-limit, limit, size=(output_dim, input_dim))
        elif weight_init == 'xavier_normal':
            std = np.sqrt(2 / (input_dim + output_dim))
            self.weights = np.random.normal(0, std, size=(output_dim, input_dim))
        else:
            raise ValueError('Weight initialization not supported')

        self.biases = np.random.randn(output_dim, 1)  # Initialize biases randomly
        self.input = None  # Store input during forward pass
        self.args = args 

        # Select optimizer based on user-specified argument
        if args.optimizer == 'sgd':
            self.optimizer = SGD(args)
        elif args.optimizer == 'momentum':
            self.optimizer = Momentum(args)
        elif args.optimizer == 'nesterov':
            self.optimizer = Nesterov(args)
        elif args.optimizer == 'adagrad':
            self.optimizer = Adagrad(args)
        elif args.optimizer == 'rmsprop':
            self.optimizer = RMSProp(args)
        elif args.optimizer == 'adam':
            self.optimizer = Adam(args)
        elif args.optimizer == 'nadam':
            self.optimizer = Nadam(args)
        else:
            raise ValueError('Optimizer not supported')

    def forward(self, x):
        """
        Performs the forward pass of the perceptron layer.
        
        Args:
            x (numpy.ndarray): Input of shape (input_dim, batch_size).
        
        Returns:
            numpy.ndarray: Output of shape (output_dim, batch_size).
        """
        self.input = x  # Store input for backward pass
        return np.dot(self.weights, x) + self.biases  # Compute linear transformation

    def backward(self, grad_output):
        """
        Computes gradients and updates weights using the selected optimizer.
        
        Args:
            grad_output (numpy.ndarray): Gradient of loss with respect to output.
        
        Returns:
            numpy.ndarray: Gradient of loss with respect to input.
        """
        grad_input = np.dot(self.weights.T, grad_output)  # Compute gradient w.r.t input
        grad_weights = np.dot(grad_output, self.input.T) / self.args.batch_size  # Compute gradient w.r.t weights
        grad_biases = np.sum(grad_output, axis=1, keepdims=True) / self.args.batch_size  # Compute gradient w.r.t biases

        # Apply L2 regularization
        grad_weights += self.args.weight_decay * self.weights
        grad_biases += self.args.weight_decay * self.biases

        # Update weights and biases using the selected optimizer
        self.weights, self.biases = self.optimizer.update(self.weights, self.biases, grad_weights, grad_biases)

        return grad_input  # Return gradient for the previous layer
