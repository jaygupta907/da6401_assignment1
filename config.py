import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Multilayer Feedforward Neural Network')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate') 
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use')
    parser.add_argument('--hidden_size', type=int, default=256, help='Number of neurons in hidden layer')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to use')
    parser.add_argument('--weight_init', type=str, default='random', help='Weight initialization')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    return parser.parse_args()