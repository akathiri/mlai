import _pickle as cPickle
import gzip
import numpy as np


def load_data():
    """
    Load MNIST data as a tuple containing the training data, the validation set and the test set
    :return: tuple of training,validation, and test sets
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


def vectorize_data():
    """
    Convert data into numpy objects
    :return: tuple of numpy objects
    """
    training_set, validation_set, test_set = load_data()

    def one_hot_vectorize(y):
        """
        Return one-hot encoded vector
        :param y: class label
        :return: numpy array
        """
        encoding = np.zeros((10, 1))
        encoding[y] = 1.0
        return encoding

    training_X = [np.reshape(x, (784, 1)) for x in training_set[0]]
    training_y = [one_hot_vectorize(y) for y in training_set[1]]

    validation_X = [np.reshape(x, (784, 1)) for x in validation_set[0]]

    test_X = [np.reshape(x, (784, 1)) for x in test_set[0]]

    return zip(training_X, training_y), zip(validation_X, validation_set[1]), zip(test_X, test_set[1])
