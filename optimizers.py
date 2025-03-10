import numpy as np


class SGD:
    def __init__(self, args):
        self.args = args

    def update(self, weights,biases,grad_weights, grad_biases):
        """
        Updates the weights and biases 

        Args:
            weights (numpy.ndarray): Current weight matrix.
            biases (numpy.ndarray): Current bias vector.
            grad_weights (numpy.ndarray): Gradient of loss w.r.t weights.
            grad_biases (numpy.ndarray): Gradient of loss w.r.t biases.

        Returns:
            tuple: Updated weights and biases.
        """
        weights -= self.args.learning_rate * grad_weights
        biases  -= self.args.learning_rate * grad_biases

        return weights,biases

class Momentum:
    def __init__(self, args):
        self.args = args
        self.v_w = 0
        self.v_b = 0

    def update(self, weights,biases,grad_weights, grad_biases):
        """
        Updates the weights and biases 

        Args:
            weights (numpy.ndarray): Current weight matrix.
            biases (numpy.ndarray): Current bias vector.
            grad_weights (numpy.ndarray): Gradient of loss w.r.t weights.
            grad_biases (numpy.ndarray): Gradient of loss w.r.t biases.

        Returns:
            tuple: Updated weights and biases.
        """
        self.v_w = self.args.momentum * self.v_w + grad_weights
        self.v_b = self.args.momentum * self.v_b + grad_biases
        weights -= self.args.learning_rate * self.v_w
        biases  -= self.args.learning_rate * self.v_b

        return weights,biases

class Nesterov:
    def __init__(self, args):
        self.args = args
        self.v_w = 0
        self.v_b = 0

    def update(self,weights,biases ,grad_weights, grad_biases):
        """
        Updates the weights and biases 

        Args:
            weights (numpy.ndarray): Current weight matrix.
            biases (numpy.ndarray): Current bias vector.
            grad_weights (numpy.ndarray): Gradient of loss w.r.t weights.
            grad_biases (numpy.ndarray): Gradient of loss w.r.t biases.

        Returns:
            tuple: Updated weights and biases.
        """
        prev_v_w = self.v_w
        prev_v_b = self.v_b


        self.v_w = self.args.momentum * self.v_w - self.args.learning_rate * grad_weights
        self.v_b = self.args.momentum * self.v_b - self.args.learning_rate * grad_biases


        weights += -self.args.momentum * prev_v_w + (1 + self.args.momentum) * self.v_w
        biases += -self.args.momentum * prev_v_b + (1 + self.args.momentum) * self.v_b

        return weights, biases

class Adagrad:
    def __init__(self, args):
        self.args = args
        self.v_w = 0
        self.v_b = 0

    def update(self,weights,biases, grad_weights, grad_biases):
        """
        Updates the weights and biases 

        Args:
            weights (numpy.ndarray): Current weight matrix.
            biases (numpy.ndarray): Current bias vector.
            grad_weights (numpy.ndarray): Gradient of loss w.r.t weights.
            grad_biases (numpy.ndarray): Gradient of loss w.r.t biases.

        Returns:
            tuple: Updated weights and biases.
        """
        self.v_w += grad_weights**2
        self.v_b += grad_biases**2
        weights -= self.args.learning_rate * grad_weights / (np.sqrt(self.v_w) + self.args.epsilon)
        biases  -= self.args.learning_rate * grad_biases / (np.sqrt(self.v_b) + self.args.epsilon)

        return weights,biases

class RMSProp:
    def __init__(self, args):
        self.args = args
        self.v_w = 0
        self.v_b = 0
    def update(self,weights,biases, grad_weights, grad_biases):
        """
        Updates the weights and biases 

        Args:
            weights (numpy.ndarray): Current weight matrix.
            biases (numpy.ndarray): Current bias vector.
            grad_weights (numpy.ndarray): Gradient of loss w.r.t weights.
            grad_biases (numpy.ndarray): Gradient of loss w.r.t biases.

        Returns:
            tuple: Updated weights and biases.
        """
        self.v_w = self.args.beta * self.v_w + (1 - self.args.beta) * grad_weights**2   
        self.v_b = self.args.beta * self.v_b + (1 - self.args.beta) * grad_biases**2
        weights -= self.args.learning_rate * grad_weights / (np.sqrt(self.v_w) + self.args.epsilon)
        biases  -= self.args.learning_rate * grad_biases / (np.sqrt(self.v_b) + self.args.epsilon)

        return weights,biases

class Adam:
    def __init__(self, args):
        self.args = args
        self.m_w = 0
        self.m_b = 0
        self.v_w = 0
        self.v_b = 0
        self.timestep = 0
    
    def update(self,weights,biases, grad_weights, grad_biases):
        """
        Updates the weights and biases 

        Args:
            weights (numpy.ndarray): Current weight matrix.
            biases (numpy.ndarray): Current bias vector.
            grad_weights (numpy.ndarray): Gradient of loss w.r.t weights.
            grad_biases (numpy.ndarray): Gradient of loss w.r.t biases.

        Returns:
            tuple: Updated weights and biases.
        """
        self.timestep += 1
        self.m_w = self.args.beta1 * self.m_w + (1 - self.args.beta1) * grad_weights
        self.m_b = self.args.beta1 * self.m_b + (1 - self.args.beta1) * grad_biases
        self.v_w = self.args.beta2 * self.v_w + (1 - self.args.beta2) * grad_weights**2
        self.v_b = self.args.beta2 * self.v_b + (1 - self.args.beta2) * grad_biases**2
        m_w_hat = self.m_w / (1 - self.args.beta1**self.timestep)
        m_b_hat = self.m_b / (1 - self.args.beta1**self.timestep)   
        v_w_hat = self.v_w / (1 - self.args.beta2**self.timestep)
        v_b_hat = self.v_b / (1 - self.args.beta2**self.timestep)
        weights -= self.args.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.args.epsilon)
        biases  -= self.args.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.args.epsilon)

        return weights,biases
    
class Nadam:
    def __init__(self, args):
        self.args = args
        self.m_w = 0
        self.m_b = 0
        self.v_w = 0
        self.v_b = 0
        self.timestep = 0

    def update(self,weights,biases, grad_weights, grad_biases):
        """
        Updates the weights and biases 

        Args:
            weights (numpy.ndarray): Current weight matrix.
            biases (numpy.ndarray): Current bias vector.
            grad_weights (numpy.ndarray): Gradient of loss w.r.t weights.
            grad_biases (numpy.ndarray): Gradient of loss w.r.t biases.

        Returns:
            tuple: Updated weights and biases.
        """
        self.timestep += 1
        self.m_w = self.args.beta1 * self.m_w + (1 - self.args.beta1) * grad_weights
        self.m_b = self.args.beta1 * self.m_b + (1 - self.args.beta1) * grad_biases
        self.v_w = self.args.beta2 * self.v_w + (1 - self.args.beta2) * grad_weights**2
        self.v_b = self.args.beta2 * self.v_b + (1 - self.args.beta2) * grad_biases**2
        m_w_hat = self.m_w / (1 - self.args.beta1**self.timestep)
        m_b_hat = self.m_b / (1 - self.args.beta1**self.timestep)
        v_w_hat = self.v_w / (1 - self.args.beta2**self.timestep)
        v_b_hat = self.v_b / (1 - self.args.beta2**self.timestep)
        weights -= self.args.learning_rate * (self.args.beta1 * m_w_hat + (1 - self.args.beta1) * grad_weights/(1-self.args.beta1**self.timestep)) / (np.sqrt(v_w_hat) + self.args.epsilon)
        biases  -= self.args.learning_rate * (self.args.beta1 * m_b_hat + (1 - self.args.beta1) * grad_biases/(1-self.args.beta1**self.timestep)) / (np.sqrt(v_b_hat) + self.args.epsilon)

        return weights,biases