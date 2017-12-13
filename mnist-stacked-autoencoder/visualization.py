import chainer
from chainer import serializers
import matplotlib.pyplot as plt
import net, data
import importlib
import numpy as np
import plot as plot



mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
print mnist
x_train, x_test = np.split(mnist['data'], [data.num_train])
x_train_label,x_test_label = np.split(mnist['target'], [data.num_train])
x_train = x_train.reshape((len(x_train), 1, 28, 28))
x_test = x_test.reshape((len(x_test), 1, 28, 28))

print x_test




model = net.Regression(net.AutoEncoder(28, 10, 2, 9, 'relu'))
serializers.load_npz('relu_9x10filters_2hidden_epoch20_alpha0.001_noise0.model', model)

var = chainer.Variable(x_test)
print var
feature = model.predictor.encoder(var)

print feature[0]
plot.scatter_labeled_z(feature, x_test_label, "scatter_z.png")
