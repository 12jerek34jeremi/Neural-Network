import random

import MNIST_loader as MNIST
from Matrix import Matrix


class Net:
    def __init__(self, shape):
        """This constructor create new Net. Argument 'shape' is tuple of number of neurons in consecutive layers.
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
            self.biases.append(Matrix(shape[i], 1, Net.creator))
            self.weights.append(Matrix(shape[i], shape[i - 1], Net.creator))

    def run(self, x):
        """This method is running the net using the given input (x) and returning an output matrix
        (an matrix of shape (n, 1), where n is natural number)"""
        result = Matrix.sigma((self.weights[1] * x) - self.biases[1])
        for weights, biases in zip(self.weights[2:], self.biases[2:]):
            result = Matrix.sigma((weights * result) - biases)
        return result

    def get_label(self, x):
        """This method is running the net using the given input (x) (by Net.run method) and returing a label.
           It return index of neuron in the last layer which has the highest activation value."""
        result = self.run(x)
        max_output = result[0][0]
        max_index = 0
        for i in range(1, 10):
            if max_output < result[i][0]:
                max_index = i
                max_output = result[i][0]
        return max_index

    def test_net(self):
        """This method run Net with all examples from MNIST test data set and returns fraction which tells how many
            images where classified correctly. (this method uses Net.get_label metod and two method from MNIST_loader
            module which are reading files."""
        accurate = 0
        for i in range(20):
            print("Starting testing images from ", i * 500, " to ", (i * 500) + 500)
            images = MNIST.load_test_images(i * 500, (i * 500) + 500)
            labels = MNIST.load_test_labels(i * 500, (i * 500) + 500)
            for j in range(500):
                if self.get_label(images[j]) == labels[j]:
                    accurate += 1
            print("Finished. Accurate: ", accurate)
        return accurate / 10000

    def epoch(self, mini_batch_size, eta):
        """ This method run an epoch. Rest is self explanatory."""
        mini_batch_indexes = [i for i in range(0, 60000, mini_batch_size)]
        mini_batch_counter = 0
        for i in range(len(mini_batch_indexes) - 1):  # in this loop are located all operations for one specific mini_batch
            inputs = MNIST.load_training_images(mini_batch_indexes[i], mini_batch_indexes[i + 1])
            outputs = MNIST.load_train_y(mini_batch_indexes[i], mini_batch_indexes[i + 1])

            overall_weights_derivatives = [None]
            # overall_weights_derivatives are sums of weights derivatives
            # from each individual example (input) in mini_batch
            # it is created in a loop beneath

            for l in range(1, self.num_of_layers):
                overall_weights_derivatives.append(Matrix(self.shape[l], self.shape[l - 1]))

            overall_biases_derivatives = [None]
            for l in range(1, self.num_of_layers):
                overall_biases_derivatives.append(Matrix(self.shape[l], 1))

            example_counter = 1
            for input, output in zip(inputs, outputs):
                weights_derivatives, biases_derivatives = self.backpropagate(input, output)
                for l in range(1, self.num_of_layers):
                    overall_weights_derivatives[l] += weights_derivatives[l]
                    overall_biases_derivatives[l] += biases_derivatives[l]
                example_counter += 1

            for l in range(1, self.num_of_layers):
                self.weights[l] -= overall_weights_derivatives[l] * (eta / mini_batch_size)
                self.biases[l] -= overall_biases_derivatives[l] * (eta / mini_batch_size)
            mini_batch_counter += 1
            if mini_batch_counter % 100 == 0:
                print("Finished minibatch number: ", mini_batch_counter)
            # end of the loop for one minibatch

    def backpropagate(self, input, y):
        """This function return all weights and biases derivatives given an input and desired output
            It is using four backpropagation equations."""

        a = [input]  # An array of all activations
        z = [None]  # An array for all z's (activation before sigma function)
        errors = [None] * self.num_of_layers # this are so called errors

        weights_derivatives = [None]  # an array of all weights derivatives, created in a loop beneath
        for i in range(1, self.num_of_layers):
            weights_derivatives.append(Matrix(self.shape[i], self.shape[i - 1]))

        for i in range(1, self.num_of_layers):
            # try:
            #     z.append((self.weights[i] * a[i - 1]) - self.biases[i])
            # except Exception:
            #     pass
            z.append((self.weights[i] * a[i - 1]) - self.biases[i])
            a.append(Matrix.sigma(z[i]))
        # in above loop I count all activations and z's

        errors[self.num_of_layers - 1] = \
            Matrix.hadamar_product(a[self.num_of_layers - 1] - y, Matrix.sigma_derivative(a[self.num_of_layers-1]))
        # in the line above I am counting an error for the last layer using first backpropagation equation

        for i in range(self.num_of_layers - 2, 0, -1):
            errors[i] = Matrix.hadamar_product(self.weights[i + 1].transpose() * errors[i + 1],
                                               Matrix.sigma_derivative(a[i]))

        # in loop above I am counting errors for all layers in the network using second backpropagation equation (BP2)

        for l in range(self.num_of_layers - 1, 0, -1):
            for j in range(self.shape[l]):
                for k in range(self.shape[l - 1]):
                    weights_derivatives[l][j][k] = a[l - 1][k][0] * errors[l][j][0]
        # in a loop above i am counting all weights derivatives using fourth backpropagation equation (BP4)

        return weights_derivatives, errors
        # returning a tuple of all weights and biases derivatives using the BP3
        # BP3 tells that specific error is equal to specific biases derivatives
        # (the ratio between them is equal to zero)

    @staticmethod
    def creator():
        # return random.randrange(0, 6)
        return random.random() * 3.0 - 1.5
