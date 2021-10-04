import numpy as np
import random
import pickle
import copy

# NOT FINISHED, class 'Neuron" is finished, this is not 

class SoftmaxNetwork:
    def __init__(self, shape, name = None):
        """This constructor create new Network. Argument 'shape' is tuple of number of neurons in consecutive layers.
        For example if shape is (100, 30, 30, 5), net will have 100 neurons in input layer (layer 0), 30 in first layer
        an so on. This constructor create list of matrices representing weights and biases and initialize with normal
        distribution.
        In this net input and x are synonyms. They mean an input matrix, it is an matrix representing input layer.
        (Shape of this matrix is (n, 1), where n is natural number.)
        Output (synonym to y) is an (desired or real) output of the network, so the matrix of shape (n, 1) where n is
        natural number.
        Label is just a digit, just a number, connected to specific image of digit. (number mean int or float)
        """
        self.num_of_layers = len(shape)
        self.biases = [None]
        self.weights = [None]
        self.shape = shape
        for i in range(1, self.num_of_layers):
            self.biases.append(np.random.normal(0.0, 1.5, (shape[i], 1)))
            self.weights.append(np.random.normal(0.0, 1.5, (shape[i], shape[i - 1])))
        if name:
            self.name = name
        else:
            self.name = "neural_net"

    def save(self, add_shape = True, name = None, file_path = "saves/"):
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
        result = (self.weights[1] @ x) - self.biases[1]
        for weights, biases in zip(self.weights[2:], self.biases[2:]):
            result = SoftmaxNetwork.sigmoid(result)
            result = (weights @ result) - biases

        result = np.exp(result, out = result)
        sum = np.sum(result)
        result /= sum
        return result

    @staticmethod
    def sigmoid(z):
         return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        "a is some layer of activations"
        return a * (1.0 - a)


    def test_net(self, test_data):
        """This method run Network with all examples of test_data (arg) and returns fraction which tells how many
            images where classified correctly. test_data is list of tupples. Each tupple is numpy matrix representing
            an image and an label connected to that image."""
        accurate = 0
        for x, label in test_data:
            if np.argmax(self.run(x)) == label:
                accurate += 1
        return accurate / len(test_data)

    def mini_batch(self, tuples, eta, mini_batch_size, cost_function, dropout = False, L1_regularization_prm = None, L2_regularization_prm = None):
        """Argument cost function tell which cost function to use.
            1 for qudratic cost function
            2 for entropy cost--function
        """

        if dropout:
            weights = copy.deepcopy(self.weights)
            for layer_weights in weights[1:]:
                n = layer_weights.shape[1]
                false_n = n // 2
                true_n = n - false_n
                index = np.concatenate((np.full(true_n, True), np.full(false_n, False)))
                np.random.shuffle(index)
                layer_weights[:, index] = 0.0
            self.weights, weights = weights, self.weights

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

        for x, y in tuples:
            weights_derivatives, biases_derivatives = self.backpropagate(x, y, cost_function)
            for l in range(1, self.num_of_layers):
                overall_weights_derivatives[l] += weights_derivatives[l]
                overall_biases_derivatives[l] += biases_derivatives[l]

        if dropout:
            self.weights, weights = weights, self.weights

        for l in range(1, self.num_of_layers):
            self.weights[l] -= overall_weights_derivatives[l] * (eta / mini_batch_size)
            self.biases[l] -= overall_biases_derivatives[l] * (eta / mini_batch_size)

    def backpropagate(self, x, y):
        """This function return all weights and biases derivatives given an input and desired output
            It is using four backpropagation equations.
            Argument cost function tell which cost function to use.
            1 for quadratic cost function
            2 for entropy cost--function
            """
        a = [x]  # An array of all activations
        z = [None]  # An array for all z's (activation before sigma function)
        errors = [None] * self.num_of_layers # this are so called errors
        weights_derivatives = [None] * self.num_of_layers # an array of all weights derivatives, created in a loop beneath


        for i in range(1, self.num_of_layers):
            z.append((self.weights[i] @ a[i - 1]) - self.biases[i])
            a.append(SoftmaxNetwork.sigmoid(z[i]))
        # in above loop I count all activations and z's

        errors[self.num_of_layers - 1] = a[self.num_of_layers-1] - y
        # in the line above I am counting an error for the last layer using first backpropagation equation

        for l in range(self.num_of_layers - 2, 0, -1):
            errors[l] = (self.weights[l+1].T @ errors[l+1]) * SoftmaxNetwork.sigmoid_derivative(a[l])
        # in loop above I am counting errors for all layers in the network using second backpropagation equation (BP2)

        for l in range(self.num_of_layers - 1, 0, -1):
                    weights_derivatives[l] = np.dot(errors[l], a[l-1].T)
        # in a loop above i am counting all weights derivatives using fourth backpropagation equation (BP4)

        return weights_derivatives, errors
        # returning a tuple of all weights and biases derivatives using the BP3
        # BP3 tells that specific error is equal to specific biases derivatives
        # (the ratio between them is equal to zero)

    def train(self, number_of_epochs, mini_batch_size, eta, cost_function, train_data, test_data = None, dropout=False,
              L1_regularization_prm = None, L2_regularization_prm = None):
        """Argument cost function tell which cost function to use.
            1 for quadratic cost function
            2 for entropy cost--function

            This function shuffle the train_data ( it alternate the list)
        """
        print("Starting training")
        n = len(train_data)
        for epoch in range(number_of_epochs):
            random.shuffle(train_data)
            for i in range(0, n, mini_batch_size):
                self.mini_batch(train_data[i : i+mini_batch_size], eta, mini_batch_size, cost_function, dropout, L1_regularization_prm, L2_regularization_prm)
            print("Finished epoch ", epoch)
            if test_data:
                print("Efficiency: ", self.test_net(test_data))
