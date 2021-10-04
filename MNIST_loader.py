import numpy as np
from PIL import Image
import pickle

def load_training_tuples():
    """This method is returning a 60'000 elements list. Each element of this list is tuple of x and y, where x
    is an input matrix representing an image, and y is desired output matrix (a matrix of shape (10, 1) ).
    IT'S NOT THE SAME as load_test_tuples. In this (load_training_tuples) method second element of a tuple is numpy
     array (matrix), while in load_test_tuples second element of a tuple is label (just a int number) """
    return list(zip(load_training_images(), load_training_outputs()))

def load_test_tuples():
    """This method is returning a 10'000 elements list. Each element of this list is tuple of x and label, where x
    is an input matrix representing an image, and label is digit (int) which is on this image.
    IT'S NOT THE SAME as load_training_tuples. In this method (load_test_tuples) second element of a tuple is label
    (just a int number), while in load_training_tuples second element of a tuple is numpy array (matrix)"""
    a = load_test_images()
    b = load_test_labels()
    return list(zip(a, b))

def show_image(x):
    image = Image.new("RGB", (28, 28))
    pixel_map = image.load()
    i = 0
    for y_index in range(28):
        for x_index in range(28):
            color = int(255 * x[i, 0])
            pixel_map[x_index, y_index] = (color, color, color)
            i += 1
    image.show()
    image.close()


def load_training_images():
    r_list = [] #return list
    with open("MNIST/train-images.idx3-ubyte", "rb") as file:
        file.seek(16)
        for i in range(60000):
            x = (np.fromfile(file, dtype=np.ubyte, count=784) / 255)
            x.resize((784, 1))
            r_list.append(x)
    return r_list

def load_test_images():
    r_list = [] #return list
    with open("MNIST/t10k-images.idx3-ubyte", "rb") as file:
        file.seek(16)
        for i in range(10000):
            x = (np.fromfile(file, dtype=np.ubyte, count=784) / 255)
            x.resize((784, 1))
            r_list.append(x)
    return r_list

def load_training_outputs():
    r_list = [np.zeros((10,1)) for i in range(60000)]
    with open("MNIST/train-labels.idx1-ubyte", "rb") as file:
        label_data = file.read()[8:]
        for i in range(60000):
            r_list[i][label_data[i],0] = 1.0
    return r_list

def load_test_outputs():
    r_list = [np.zeros((10,1)) for i in range(10000)]
    with open("MNIST/t10k-labels.idx1-ubyte", "rb") as file:
        label_data = file.read()[8:]
        for i in range(10000):
            r_list[i][label_data[i],0] = 1.0
    return r_list

def load_training_labels():
    with open("MNIST/train-labels.idx1-ubyte", "rb") as file:
        label_data = file.read()
        r_list = [label_data[i] for i in range(8, 60008)]
    return r_list

def load_test_labels():
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

    def show_image_and_label(self, index):
        x, y = self.load_tupple(index)
        label = self.load_label(index)
        image = Image.new("RGB", (28, 28))
        pixel_map = image.load()
        i = 0
        for y_index in range(28):
            for x_index in range(28):
                color = int(255 * x[i, 0])
                pixel_map[x_index, y_index] = (color, color, color)
                i += 1
        image.show()
        print("desired_output_matrix: ", y)
        print("desired_label: ", label)
        image.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.image_file.close()
