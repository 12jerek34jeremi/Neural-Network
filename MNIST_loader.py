from Matrix import Matrix
from PIL import Image


def load_test_images(start, stop):
    """This function is reading appropriete file and returning a list of matrices,
       representing specific image. Argument 'start' is index of first red image
       and argument 'stop' is index of first not-red image.
       Shape of returned matrix is (784, 1). (an image is square of 28*28 = 784 pixels)
       This functions is using, of course, MNIST training data set."""
    file = open('MNIST/t10k-images.idx3-ubyte', 'rb')
    data = file.read()
    images = [Matrix(784, 1) for i in range(stop - start)]
    i = 16 + (start * 784)
    for photo in range(stop - start):
        for pixel in range(784):
            images[photo][pixel][0] = data[i] / 255
            i += 1
    return images


def load_training_images(start, stop):
    """This function is reading appropriete file and returning a list of matrices,
       representing specific image. Argument 'start' is index of first red image
       and arguments 'stop' is index of first not-red image.
       Shape of returned matrix is (784, 1). (an image is square of 28*28 = 784 pixels)
       This functions is using, of course, MNIST test data set.
       This matrix is then use as first layer activations (input layer)"""
    file = open('MNIST/train-images.idx3-ubyte', 'rb')
    data = file.read()
    images = [Matrix(784, 1, lambda: 0) for i in range(stop - start)]
    i = 16 + (start * 784)
    for photo in range(stop - start):
        for pixel in range(784):
            images[photo][pixel][0] = data[i] / 255
            i += 1
    return images


def load_training_labels(start, stop):
    """ This method is reading appropriate file and returning list of labels
        connected to a specific image. An item of a list is just a number.
        Argument 'start' is index of first red label and argument 'stop'
        is index of first not-read label.
        This functions is using, of course, MNIST training data set."""
    file = open('MNIST/train-labels.idx1-ubyte', 'rb')
    data = file.read()
    labels = [0.0 for i in range(stop - start)]
    i = 8 + start
    for label in range(stop - start):
        labels[label] = data[i]
        i += 1
    return labels


def load_test_labels(start, stop):
    """ This method is reading appropriate file and returning list of labels
        connected to a specific image. An item of a list is just a number.
        Argument 'start' is index of first red label and argument 'stop'
        is index of first not-read label.
        This functions is using, of course, MNIST training test set."""

    file = open('MNIST/t10k-labels.idx1-ubyte', 'rb')
    data = file.read()
    labels = [0.0 for i in range(stop - start)]
    i = 8 + start
    for label in range(stop - start):
        labels[label] = data[i]
        i += 1
    return labels


def load_train_y(start, stop):
    """ This method is reading appropriate file and returning
        list of desired output matrices.
        For example if on image is digit '4' it returns such a matrix:
        [[0.0]
        [0.0]
        [0.0]
        [0.0]
        [1.0]
        [0.0]
        [0.0]
        [0.0]
        [0.0]
        [0.0]]
        Argument 'start' is index of first red label and argumnet 'stop'
        is index of first not-red label.
        This method is, of course, using MNIST training data set."""
    result = [Matrix(10, 1) for i in range(0, stop - start)]
    labels = load_training_labels(start, stop)

    for matrix, label in zip(result, labels):
        matrix[label][0] = 1.0
    return result


def show_label_and_image(index, mode):
    """This method is reading appropriate files and then showing image
     of given index (see arguments list) and writing in console label
     connected to that image. If value of "mode" (see arguments list) is
     'training' method is using MNIST training data set, if value of mode
     is 'test' method is using MNIST test data set.
    """

    image_matrix, label = None, None
    if mode == "training":
        image_matrix = load_training_images(index, index + 1)[0]
        label = load_training_labels(index, index + 1)[0]
    elif mode == "test":
        image_matrix = load_test_images(index, index + 1)[0]
        label = load_test_labels(index, index + 1)[0]
    else:
        print("Uknown mode: ", mode)
        return

    image = Image.new("RGB", (28, 28))
    pixel_map = image.load()
    index = 0
    for y in range(28):
        for x in range(28):
            color = int(255 * image_matrix[index][0])
            pixel_map[x, y] = (color, color, color)
            index += 1
    image.show()
    print("Label: ", label)
