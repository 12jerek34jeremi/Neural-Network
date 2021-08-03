from Net import Net
from MNIST_loader import Loader

my_net = Net((784, 100, 100, 10))
efficiency = my_net.test_net()
print("Efficiency: ", efficiency)
my_net.epoch(20, 3.0, 25)
efficiency = my_net.test_net()
print("Efficiency: ", efficiency)
