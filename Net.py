import random

import MNIST_loader as MNIST
from Matrice import Matrice

class Net:
    def __init__(self, shape):
        self.num_of_layers = len(shape)
        self.biases = [None]
        self.weights = [None]
        self.shape = shape
        for i in range(1, self.num_of_layers):
            self.biases.append(Matrice(shape[i], 1, Net.creator))
            self.weights.append(Matrice(shape[i], shape[i-1], Net.creator))

    def run(self, x):
        result = Matrice.sigma((self.weights[1] * x ) - self.biases[1])
        for weights, biases in zip(self.weights[2:], self.biases[2:]):
            result = Matrice.sigma((weights * result) - biases)
        return result

    def get_label(self, x):
        result = self.run(x)
        max_output = result[0][0]
        max_index = 0
        for i in range(1, 10):
            if(max_output < result[i][0]):
                max_index = i
                max_output = result[i][0]
        return max_index

    def test_net(self):
        acurate = 0
        for i in range(20):
            print("Starting testing images from ", i*500, " to ", (i*500)+500)
            images = MNIST.load_test_images(i*500, (i*500)+500)
            labels = MNIST.load_test_labels(i*500, (i*500)+500)
            for j in range(500):
                if(self.get_label(images[j]) == labels[j][0]):
                    acurate += 1
            print("Finished. Acurete: ", acurate)
        return acurate / 10000

    def epoch(self, mini_batch_size, eta):
        mini_batch_indexes = [i for i in range(0, 60000, mini_batch_size)]
        for i in range(len(mini_batch_indexes)-1):
            inputs = MNIST.load_training_images(mini_batch_indexes[i], mini_batch_indexes[i+1])
            labels = MNIST.load_training_labels(mini_batch_indexes[i], mini_batch_indexes[i+1])
            for input, label in zip(inputs, labels):
                weights_derivaties, biases_derivaties = self.backpropagate(input, label)
                # finished here





    def backpropagate(self, input, y):
        "This function return all weights and biases derivatis given an input and desired output"

        a = [input] # An array of all activations
        z = [None] #An array for all z's (activation before sigma funcction)
        errors = [ None for i in range(self.num_of_layers)]  #thies are so called errors

        weights_derivativies = [None] # an array of all weiths derivaties, created in a loop beneath
        for i in range(1, self.num_of_layers):
            weights_derivativies.append([])

        for i in range(1, self.num_of_layers):
            a.append( (self.weights[i] * a[i-1]) - self.biases[i])
            z.append( Matrice.sigma(a[i]))
        #in above loop I count all activations and z's

        errors[self.num_of_layers-1] = Matrice.hadamar_product(input - y, Matrice.sigma(z[self.num_of_layers - 1]))
        # in the line above I am counting an error for the last layer using first backpropagation equation

        for i in range(self.num_of_layers-2, 0, 0):
            errors[i] = Matrice.hadamar_product(self.weights[i+1] * errors[i+1], Matrice.sigma(z[i]))
        #in loop above I am counting errors for all layers in the network using second backpropagation equation (BP2)

        for l in range(self.num_of_layers-1, 1, -1):
            weights_derivativies[l] = [ [ 0 for j in range(self.shape[l-1])] for i in range(self.shape[l])]
            for j in range(self.shape[l]):
                for k in range(self.shape[l-1]):
                    weights_derivativies[l][j][k] = a[l-1][k][0] * errors[l][j][0]
        #in a loop above i am counting all weights derivaties using fourth backpropagation equation (BP4)

        return (weights_derivativies, errors)
        #returning a tupple of all weights and biases deriaties using the BP3


    @staticmethod
    def creator():
        # return random.randrange(0, 6)
        return random.random() * 3.0 - 1.5