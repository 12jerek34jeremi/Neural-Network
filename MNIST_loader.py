from Matrice import Matrice
from PIL import Image

def load_test_images(start, stop):
    file = open('MNIST/t10k-images.idx3-ubyte', 'rb')
    data = file.read()
    images = [ Matrice(784, 1, lambda : 0) for i in range(stop - start) ]
    i = 16 + (start * 784)
    for photo in range(stop - start):
        for pixel in range(784):
            images[photo][pixel][0] = data[i] / 255
            i += 1
    return images

def load_training_images(start, stop):
    file = open('MNIST/train-images.idx3-ubyte', 'rb')
    data = file.read()
    images = [ Matrice(784, 1, lambda : 0) for i in range(stop - start) ]
    i = 16 + (start * 784)
    for photo in range(stop - start):
        for pixel in range(784):
            images[photo][pixel][0] = data[i] / 255
            i += 1
    return images

def load_training_labels(start, stop):
    file = open('MNIST/train-labels.idx1-ubyte', 'rb')
    data = file.read()
    labels = [ [] for i in range(stop - start) ]
    i = 8 + start
    for label in range(stop - start):
        labels[label].append(data[i])
        i += 1
    return labels

def load_test_labels(start, stop):
    file = open('MNIST/t10k-labels.idx1-ubyte', 'rb')
    data = file.read()
    labels = [ [] for i in range(stop - start) ]
    i = 8 + start
    for label in range(stop - start):
        labels[label].append(data[i])
        i += 1
    return labels

def show_label_and_image(index, mode):
    image_matrice, label = None, None
    if mode == "training":
        image_matrice = load_training_images(index, index+1)[0]
        label = load_training_labels(index, index+1)[0]
    elif mode == "test":
        image_matrice = load_test_images(index, index+1)[0]
        label = load_test_labels(index, index+1)[0]
    else:
        print("Uknown mode: ", mode)
        return

    image = Image.new("RGB", (28, 28))
    pixel_map = image.load()
    index = 0
    for y in range(28):
        for x in range(28):
            color = int(255 * image_matrice[index][0])
            pixel_map[x, y] = (color, color, color)
            index += 1
    image.show()
    print("Label: ", label)

