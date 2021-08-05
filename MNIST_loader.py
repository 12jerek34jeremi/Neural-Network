import numpy as np
from PIL import Image

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
