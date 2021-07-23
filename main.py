from Net import Net

my_net = Net((784, 100, 100, 10))
my_net.test_net()
my_net.epoch(20, 3.0)
my_net.test_net()




