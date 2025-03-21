from keras._tf_keras.keras.datasets import mnist, fashion_mnist 
import numpy as np  


class Batch_Dataset:
    """
    A class to represent a dataset with batching and validation split.
    """
    def __init__(self, dataset, validation_split=0.1):
        """
        Initializes the dataset, loads the data, normalizes it, and splits it into training, validation, and test sets.
        
        Args:
            dataset (str): The name of the dataset ('mnist' or 'fashion_mnist').
            validation_split (float): The proportion of training data to be used for validation.
        """
        if dataset == 'mnist':
            (x_train, y_train), (self.x_test, self.y_test) = mnist.load_data()
            self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        elif dataset == 'fashion_mnist':
            (x_train, y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
            self.classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        else:
            raise ValueError('Dataset not supported')

        # Flatten the image data into vectors
        x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2])
        self.x_test = self.x_test.reshape(-1, self.x_test.shape[1] * self.x_test.shape[2])
        
        # Normalize the pixel values to the range [0,1]
        x_train = x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
        
        # Convert labels to one-hot encoding
        self.num_classes = len(np.unique(y_train))
        y_train = np.eye(self.num_classes)[y_train]
        self.y_test = np.eye(self.num_classes)[self.y_test]
        
        # Shuffle training data and split into training and validation sets
        num_train = len(x_train)
        indices = np.arange(num_train)
        np.random.shuffle(indices)
        
        val_size = int(num_train * validation_split)
        self.x_val, self.y_val = x_train[indices[:val_size]], y_train[indices[:val_size]]
        self.x_train, self.y_train = x_train[indices[val_size:]], y_train[indices[val_size:]]

    def create_batches(self, data_x, data_y, batch_size=32, shuffle=True):
        """
        Splits data into batches of the specified size.
        
        Args:
            data_x (numpy.ndarray): Input data.
            data_y (numpy.ndarray): Corresponding labels.
            batch_size (int): The size of each batch.
            shuffle (bool): Whether to shuffle the data before batching.

        Returns:
            list: A list of batches containing (data_x, data_y).
        """
        num_datapoints = len(data_x)
        indices = np.arange(num_datapoints)
        batches = []
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, num_datapoints, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            batches.append([data_x[batch_indices], data_y[batch_indices]])
        return batches
    
    def create_train_batches(self, batch_size=32, shuffle=True):
        """
        Creates batches from the training dataset.
        
        Args:
            batch_size (int): The size of each batch.
            shuffle (bool): Whether to shuffle the data before batching.

        Returns:
            list: A list of training batches.
        """
        return self.create_batches(self.x_train, self.y_train, batch_size, shuffle)

    def create_val_batches(self, batch_size=32, shuffle=True):
        """
        Creates batches from the validation dataset.
        
        Args:
            batch_size (int): The size of each batch.
            shuffle (bool): Whether to shuffle the data before batching.

        Returns:
            list: A list of validation batches.
        """
        return self.create_batches(self.x_val, self.y_val, batch_size, shuffle)

    def create_test_batches(self, batch_size=32, shuffle=True):
        """
        Creates batches from the test dataset.
        
        Args:
            batch_size (int): The size of each batch.
            shuffle (bool): Whether to shuffle the data before batching.

        Returns:
            list: A list of test batches.
        """
        return self.create_batches(self.x_test, self.y_test, batch_size, shuffle)
