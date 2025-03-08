import numpy as np  # Import NumPy for numerical operations



def loss_derivative_wrt_activation(y_pred, loss_grad):
    """
    Computes the gradient of loss w.r.t. softmax input (logits).

    Args:
        y_pred (numpy.ndarray): Softmax output, shape (batch_size, num_classes).
        loss_grad (numpy.ndarray): Gradient of loss w.r.t softmax output.

    Returns:
        numpy.ndarray: Gradient of loss w.r.t logits.
    """
    batch_size, num_classes = y_pred.shape
    grad_z = np.zeros_like(y_pred)

    for i in range(batch_size):
        s = y_pred[i].reshape(-1, 1)  # Convert softmax output to column vector
        jacobian = np.diagflat(s) - np.dot(s, s.T)  # Compute softmax Jacobian
        grad_z[i] = np.dot(jacobian, loss_grad[i])  # Apply chain rule

    return grad_z.T


def cross_entropy_loss(y_pred, y_true):
    """
    Computes the cross-entropy loss.
    
    Args:
        y_pred (numpy.ndarray): Predictions, shape (batch_size, num_classes).
        y_true (numpy.ndarray): True labels, shape (batch_size, num_classes).

    Returns:
        float: Loss value.
    """
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]  # Adding small value (1e-9) to avoid log(0)

def cross_entropy_derivative(y_pred, y_true):
    """
    Computes the derivative of the cross-entropy loss w.r.t. the output.

    Args:
        y_pred (numpy.ndarray): Predictions (e.g., softmax output), shape (batch_size, num_classes).
        y_true (numpy.ndarray): True labels, shape (batch_size, num_classes).

    Returns:
        numpy.ndarray: Gradient of the loss w.r.t. output.
    """
    return -y_true / (y_pred + 1e-9)  # Avoid division by zero

def mean_squared_loss(y_pred, y_true):
    """
    Computes the mean squared error (MSE) loss.
    
    Args:
        y_pred (numpy.ndarray): Predictions, shape (batch_size, num_classes).
        y_true (numpy.ndarray): True labels, shape (batch_size, num_classes).

    Returns:
        float: Mean squared loss value.
    """
    return np.sum((y_pred - y_true) ** 2) / y_true.shape[0] 


def mse_derivative(y_pred,y_true):
    """
    Computes the derivative of the mean squared error (MSE) loss.
    
    Args:
        y_pred (numpy.ndarray): Softmax output, shape (batch_size, num_classes).
        y_true (numpy.ndarray): True labels, shape (batch_size, num_classes).

    Returns:
        numpy.ndarray: Gradient of loss w.r.t softmax output.
    """
    return (2 / y_true.shape[0]) * (y_pred - y_true)