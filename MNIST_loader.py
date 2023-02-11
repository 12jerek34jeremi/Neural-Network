import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_training_tuples() -> list[tuple[np.ndarray, int]]:
    """This method is returning a 60'000 elements list. Each element of this list is tuple of x and y, where x
    is an input matrix representing an image, and y is digit (int) assisiated with that matrix).
    MNIST dataset defines two subsets: train (60 000 examples) and test (10 000). This method loads all examples from
    train set.
    """
    return list(zip(load_training_images(), load_training_labels()))

def load_test_tuples() -> list[tuple[np.ndarray, int]]:
    """This method is returning a 10'000 elements list. Each element of this list is tuple of x and label, where x
    is an input matrix representing an image, and label is digit (int) which is on this image.
    MNIST dataset defines two subsets: train (60 000 examples) and test (10 000). This method loads all examples from
    test set. """
    a = load_test_images()
    b = load_test_labels()
    return list(zip(a, b))

def show_image(x: np.ndarray):
    """
    x is numpy array of shape (784, 1) representing an image of a digit.
    """
    plt.imshow(np.reshape(x, (28,28)), vmin=0.0, vmax=1.0, cmap='gray')


def load_training_images() -> list[np.ndarray]:
    r_list = [] #return list
    with open("MNIST/train-images.idx3-ubyte", "rb") as file:
        file.seek(16)
        for i in range(60000):
            x = (np.fromfile(file, dtype=np.ubyte, count=784) / 255)
            x.resize((784, 1))
            r_list.append(x)
    return r_list

def load_test_images() -> list[np.ndarray] :
    r_list = [] #return list
    with open("MNIST/t10k-images.idx3-ubyte", "rb") as file:
        file.seek(16)
        for i in range(10000):
            x = (np.fromfile(file, dtype=np.ubyte, count=784) / 255)
            x.resize((784, 1))
            r_list.append(x)
    return r_list

def load_training_outputs() -> list[np.ndarray]:
    r_list = [np.zeros((10,1)) for i in range(60000)]
    with open("MNIST/train-labels.idx1-ubyte", "rb") as file:
        label_data = file.read()[8:]
        for i in range(60000):
            r_list[i][label_data[i],0] = 1.0
    return r_list

def load_test_outputs() -> list[np.ndarray]:
    r_list = [np.zeros((10,1)) for i in range(10000)]
    with open("MNIST/t10k-labels.idx1-ubyte", "rb") as file:
        label_data = file.read()[8:]
        for i in range(10000):
            r_list[i][label_data[i],0] = 1.0
    return r_list

def load_training_labels() -> list[int]:
    with open("MNIST/train-labels.idx1-ubyte", "rb") as file:
        label_data = file.read()
        r_list = [label_data[i] for i in range(8, 60008)]
    return r_list

def load_test_labels() -> list[int]:
    with open("MNIST/t10k-labels.idx1-ubyte", "rb") as file:
        label_data = file.read()
        r_list = [label_data[i] for i in range(8, 10008)]
    return r_list



class Loader:
    def __init__(self, mode):
        if mode == "training":
            self.image_file = open('MNIST/train-images.idx3-ubyte', 'rb')
            label_file = open('MNIST/train-labels.idx1-ubyte', 'rb')
        elif mode == "test":
            self.image_file = open('MNIST/t10k-images.idx3-ubyte', 'rb')
            label_file = open('MNIST/t10k-labels.idx1-ubyte', 'rb')

        self.label_data = label_file.read()
        label_file.close()
        self.image_file.seek(16)

    def load_x(self, index):
        x = (np.fromfile(self.image_file, dtype=np.ubyte, count=784, offset=index * 784) / 255)
        x.resize((784, 1))
        self.image_file.seek(16)
        return x

    def load_y(self, index):
        y = np.zeros((10, 1))
        y[self.label_data[8 + index], 0] = 1.0
        return y

    def load_tupple(self, index):
        return self.load_x(index), self.load_y(index)

    def __del__(self):
        self.image_file.close()

    def load_label(self, index):
        return self.label_data[8 + index]


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.image_file.close()
