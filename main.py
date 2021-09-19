from Network import Network
from SoftmaxNetwork import SoftmaxNetwork
import MNIST_loader as MNIST
import numpy as np

np.set_printoptions(suppress=True)

train_data = MNIST.load_training_tuples()
test_data = MNIST.load_test_tuples()

kinga = SoftmaxNetwork((784, 120, 10))
kinga.traingit 