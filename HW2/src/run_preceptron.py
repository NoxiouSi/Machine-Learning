from src.Perceptron import Perceptron
import numpy as np

data = np.genfromtxt('../data/perceptronData.txt')


def normalize(X, Y):
    _X = np.array(X)
    _Y = np.array(Y)

    mean = _X.mean(axis=0)
    _X -= mean
    var = _X.var()
    _X /= var

    _Y = _Y.reshape((_X.shape[0],1))

    return _X, _Y


m = data.shape[0]

X = data[:, :-1]
Y = data[:, -1]

X, Y = normalize(X, Y)

perc = Perceptron()
perc.train(X, Y, 1)
perc.print_normalized_weight()
