import numpy as np


class LogisticRegressionGD(object):

    def __init__(self):
        self._mean = None
        self._variant = None
        self._theta = None

    def _normalize(self, X, Y):
        _X = np.array(X)
        m = _X.shape[0]
        self._mean = np.mean(_X, axis=0)
        _X = _X - self._mean
        self._variant = (np.max(_X, axis=0) - np.min(_X, axis=0)) / 2
        _X = _X / self._variant
        _X = np.append(np.ones((m, 1)), _X, axis=1)
        _Y = Y.reshape(m, 1)
        return _X, _Y

    def _re_normalize(self, X):
        _X = np.array(X)
        _X = _X - self._mean
        _X = _X / self._variant
        _X = np.append(np.ones((_X.shape[0], 1)), _X, axis=1)
        return _X

    def train(self, X, Y, lr):
        _X, _Y = self._normalize(X, Y)
        m = _X.shape[0]
        lr = lr
        self._theta = np.random.rand(_X.shape[1], 1)
        diff = np.array([1])
        while diff.T.dot(diff) ** 0.5 > 0.0001:
            diff = lr * _X.T.dot(_Y - 1.0 / (1.0 + np.exp(-_X.dot(self._theta)))) / m
            self._theta = self._theta + diff
            # lr *= 0.999

    def run(self, X):
        _X = self._re_normalize(X)
        return 1.0 / (1.0 + np.exp(-_X.dot(self._theta)))
