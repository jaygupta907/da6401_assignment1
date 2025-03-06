import numpy as np
from optimizers import SGD,Momentum,Nesterov,Adagrad,RMSProp,Adam,Nadam

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
        lambda x: np.exp(x - np.max(x, axis=0, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=0, keepdims=True)), axis=0, keepdims=True),
        lambda x: np.ones_like(x)  
    )
}


class Perceptron_Layer:
    '''
    A class to represent a perceptron layer.
    '''
    def __init__(self, input_dim,output_dim,args,weight_init='random'):
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
        self.args = args

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
        '''
        Forward pass of the perceptron layer.

        Args:
            x (numpy.ndarray): Input, shape (input_dim, batch_size).

        Returns:
            numpy.ndarray: Output, shape (output_dim, batch_size).
        '''
        self.input = x  
        return np.dot(self.weights, x) + self.biases
    
    def backward(self, grad_output):
        """
        Computes and updates weights using the selected optimizer.

        Args:
            grad_output (numpy.ndarray): Gradient of loss w.r.t output, shape (out_dim, batch_size).

        Returns:
            numpy.ndarray: Gradient of loss w.r.t input, shape (in_dim, batch_size).
        """
        grad_input = np.dot(self.weights.T, grad_output)
        grad_weights = np.dot(grad_output, self.input.T) / self.args.batch_size  
        grad_biases = np.sum(grad_output, axis=1, keepdims=True) / self.args.batch_size 
        grad_weights += self.args.weight_decay * self.weights
        grad_biases  += self.args.weight_decay * self.biases 

        self.weights,self.biases = self.optimizer.update(self.weights,
                                                    self.biases,
                                                    grad_weights,
                                                    grad_biases)
        return grad_input


        


