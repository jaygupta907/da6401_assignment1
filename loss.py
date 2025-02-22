import numpy as np
import wandb

def cross_entropy_loss(y_pred, y_true):
    """
    Computes the cross-entropy loss.

    Args:
        y_pred (numpy.ndarray): Predictions, shape (batch_size, num_classes).
        y_true (numpy.ndarray): True labels, shape (batch_size, num_classes).

    Returns:
        float: Loss.
    """
    return -np.sum(y_true * np.log(y_pred+1e-9)) / y_true.shape[0]

def cross_entropy_derivative(y_pred, y_true):
    """
    Computes the derivative of the cross-entropy loss. 
    """
    return (y_pred - y_true).T