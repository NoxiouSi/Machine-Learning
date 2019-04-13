import numpy as np


class Perceptron:

    def __int__(self):
        self._theta = None

    def train(self, X, Y, lr):
        _X = np.array(X)
        m, d = X.shape
        _Y = np.array(Y).reshape(m,1)
        _X = np.append(np.ones((m,1)), _X, axis=1)
        d += 1
        lr /= m
        self._theta = np.random.random((d, 1))
        miss = np.logical_xor(_X.dot(self._theta) > 0, Y)
        it = 0
        while miss.sum() > 0:
            it += 1
            diff = lr * _X[miss.flatten()].sum(axis=0)
            self._theta += diff.reshape(d,1)
            miss = np.logical_xor(_X.dot(self._theta) > 0, Y)
            print('iteration', it, ', total mistake:', miss.sum())

    def run(self, X):
        _X = np.array(X)
        return _X.dot(self._theta) > 0

    def print_normalized_weight(self):
        print('Classifier weights:', self._theta.flatten())
        t = self._theta.flatten()
        print('Normalized with threshold:', t[1:] / -t[:1])
