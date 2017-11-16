import numpy as np
import random


def sigmoid(z):
    """
    Sigmoid activation function
    :param z: input vector
    :return: vector with sigmoid applied
    """
    return 1.0 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of sigmoid activation function
    :param z: input vector
    :return: sigmoid prime
    """
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes):
        """
        Create internal parameters
        :param sizes: list defining the structure of neural network
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Return output of the network
        :param a: input vector
        :return: output vector
        """
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a

    def mse_cost_function(self, X, y):
        """
        compute the MSE cost
        :param X: input vector
        :param y: true output
        :return: cost of how bad the network predicts
        """
        return 0.5 * (sum(self.feedforward(X) - y) ** 2)

    @staticmethod
    def mse_cost_function_prime(output_activations, y):
        """
        compute derivation of MSE cost function
        :param output_activations: predicted output
        :param y: true output
        :return: derivative vector
        """
        return output_activations - y

    def backprop(self, X, y):
        """
        Return a tuple (nabla_b, nabla_w) representing the partial gradient of cost function
        w.r.t. biases and weights.
        :param X: input vector
        :param y: true output vector
        :return: tuple
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feed forward
        activation = X
        activations = [activation]  # list to store all activations
        zs = []  # list to store all z vectors
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # reverse-mode differentiation
        delta = self.mse_cost_function_prime(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # reverse looping
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].T, delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].T)
        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
        """
        Update network's hyperparameters by applying gradient descent
        :param mini_batch: list of tuples (X, y)
        :param eta: learning rate
        :return: None
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update rule
        self.weights = [w - eta / len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta / len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train neural network using mini-batch stochastic gradient descent
        :param training_data: list of tuples (x, y)
        :param epochs: number of iterations
        :param mini_batch_size: batch size
        :param eta: learning rate
        :param test_data: if test_data is provided then evaluate
        :return: None
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)

            # create mini batches out of training sample
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print('Epoch {0} : {1} / {2}'.format(i, self.evaluate(test_data), n_test))

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
