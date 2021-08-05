import numpy as np
import random
from MNIST_loader import Loader
import pickle



class Network:
    def __init__(self, shape, name = None):
        """This constructor create new Network. Argument 'shape' is tuple of number of neurons in consecutive layers.
        For example if shape is (100, 30, 30, 5), net will have 100 neurons in input layer (layer 0), 30 in first layer
        an so on. This constructor create list of matrices representing weights and biases and initialize them with
        'creator' method of this class.
        In this net input and x are synonyms. They mean an input matrix, it is an matrix representing input layer.
        (Shape of this matrix is (n, 1), where n is natural number.)
        Output is an (desired or real) output of the network, so the matrix of shape (n, 1) where n is natural number.
        Label is just a digit, just a number, connected to specific image of digit. (number mean int or float)
        """
        self.num_of_layers = len(shape)
        self.biases = [None]
        self.weights = [None]
        self.shape = shape
        for i in range(1, self.num_of_layers):
            self.biases.append(np.random.normal(0.0, 1.5, (shape[i], 1)))
            self.weights.append(np.random.normal(0.0, 1.5, (shape[i], shape[i-1])))
        if name:
            self.name = name
        else:
            self.name = "neural_net"

    def save(self, add_shape = True, name = None):
        file_path = "saves/"
        if name:
            file_path += name
        else:
            file_path += self.name
        if add_shape:
            file_path += "_shape" + self.shape.__str__()
        file_path += ".network"

        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_name, directory = "saves/"):
        file_name = directory + file_name
        with open(file_name, "rb") as file:
            network = pickle.load(file)
        return network

    def run(self, x):
        """This method is running the net using the given input (x) and returning an output matrix
        (an matrix of shape (n, 1), where n is natural number)"""
        result = Network.sigma((self.weights[1] @ x) - self.biases[1])
        for weights, biases in zip(self.weights[2:], self.biases[2:]):
            result = Network.sigma((weights @ result) - biases)
        return result

    @staticmethod
    def sigma(matrix):
         return 1 / (1 + np.exp(-matrix))

    def get_label(self, x):
        """This method is running the net using the given input (x) (by Network.run method) and returing a label.
           It return index of neuron in the last layer which has the highest activation value."""
        result = self.run(x)
        max_output = result[0, 0]
        max_index = 0
        for i in range(1, 10):
            if max_output < result[i, 0]:
                max_index = i
                max_output = result[i][0]
        return max_index

    def test_net(self):
        """This method run Network with all examples from MNIST test data set and returns fraction which tells how many
            images where classified correctly. (this method uses Network.get_label metod and two method from MNIST_loader
            module which are reading files."""
        accurate = 0
        with Loader("test") as loader:
            images = [loader.load_x(i) for i in range(10000)]
            labels = [loader.load_label(i) for i in range(10000)]
        for x, label in zip(images, labels):
            if self.get_label(x) == label:
                accurate += 1
        return accurate / 10000

    def epoch(self, mini_batch_size, eta):
        """ This method run an epoch. Rest is self explanatory."""

        with Loader("training") as loader:
            tuples = [(loader.load_x(i), loader.load_y(i)) for i in range(60000)]
        random.shuffle(tuples)
        for i in range(0, 60000, mini_batch_size):
            self.mini_batch(tuples[i : i+mini_batch_size], eta, mini_batch_size)

    def mini_batch(self, tupples, eta, mini_batch_size):
        overall_weights_derivatives = [None]
        # overall_weights_derivatives are sums of weights derivatives
        # from each individual example (input) in mini_batch
        # it is created in a loop beneath
        for l in range(1, self.num_of_layers):
            overall_weights_derivatives.append(np.zeros((self.shape[l], self.shape[l-1])))

        overall_biases_derivatives = [None]
        # overall_weights_derivatives are sums of bias derivatives
        # from each individual example (input) in mini_batch
        # it is created in a loop beneath
        for l in range(1, self.num_of_layers):
            overall_biases_derivatives.append(np.zeros((self.shape[l], 1)))

        for x, y in tupples:
            weights_derivatives, biases_derivatives = self.backpropagate(x, y)
            for l in range(1, self.num_of_layers):
                overall_weights_derivatives[l] += weights_derivatives[l]
                overall_biases_derivatives[l] += biases_derivatives[l]

        for l in range(1, self.num_of_layers):
            self.weights[l] -= overall_weights_derivatives[l] * (eta / mini_batch_size)
            self.biases[l] -= overall_biases_derivatives[l] * (eta / mini_batch_size)

    def backpropagate(self, x, y):
        """This function return all weights and biases derivatives given an input and desired output
            It is using four backpropagation equations."""
        a = [x]  # An array of all activations
        z = [None]  # An array for all z's (activation before sigma function)
        errors = [None] * self.num_of_layers # this are so called errors

        weights_derivatives = [None] * self.num_of_layers # an array of all weights derivatives, created in a loop beneath
        # for i in range(1, self.num_of_layers):
        #     weights_derivatives.append(np.empty((self.shape[i], self.shape[i - 1]), dtype=float))


        for i in range(1, self.num_of_layers):
            z.append((self.weights[i] @ a[i - 1]) - self.biases[i])
            a.append(Network.sigma(z[i]))
        # in above loop I count all activations and z's

        errors[self.num_of_layers - 1] = \
            (a[self.num_of_layers - 1] - y) * Network.sigma_derivative(a[self.num_of_layers - 1])
        # in the line above I am counting an error for the last layer using first backpropagation equation

        for l in range(self.num_of_layers - 2, 0, -1):
            errors[l] = (self.weights[l+1].T @ errors[l+1]) * Network.sigma_derivative(a[l])
        # in loop above I am counting errors for all layers in the network using second backpropagation equation (BP2)

        for l in range(self.num_of_layers - 1, 0, -1):
                    weights_derivatives[l] = np.dot(errors[l], a[l-1].T)
        # in a loop above i am counting all weights derivatives using fourth backpropagation equation (BP4)

        return weights_derivatives, errors
        # returning a tuple of all weights and biases derivatives using the BP3
        # BP3 tells that specific error is equal to specific biases derivatives
        # (the ratio between them is equal to zero)

    @staticmethod
    def sigma_derivative(a):
        "a is some layer of activations"
        return a * (np.ones_like(a) - a)

    def train(self, number_of_epochs, mini_batch_size, eta, test=True):
        print("Starting training")
        for epoch in range(number_of_epochs):
            self.epoch(mini_batch_size, eta)
            print("Finished epoch ", epoch)
            if test:
                print("Efficiency: ", self.test_net())
