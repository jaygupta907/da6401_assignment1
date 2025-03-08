import argparse  # Import the argparse module for handling command-line arguments

def get_args():
    """
    Parses command-line arguments and returns them.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Multilayer Feedforward Neural Network')

    # Adding various arguments for training configuration
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate') 
    parser.add_argument('--dataset', type=str, default='fashion_mnist', help='Dataset to use')
    parser.add_argument('--hidden_size', type=int, default=64, help='Number of neurons in hidden layer')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to use')
    parser.add_argument('--weight_init', type=str, default='xavier_normal', help='Weight initialization method')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 parameter for Adam optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon value for Adam optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for momentum-based optimizers')
    parser.add_argument('--weight_decay', type=float, default=0.000, help='L2 regularization weight decay')
    parser.add_argument('--wandb_project', type=str, default='Multilayer_FeedForward_Network', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='jay_gupta-indian-institute-of-technology-madras', help='WandB entity name')
    parser.add_argument('--beta', type=float, default=0.9, help='Beta parameter for RMSProp optimizer')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='Loss function to use')

    return parser.parse_args()
