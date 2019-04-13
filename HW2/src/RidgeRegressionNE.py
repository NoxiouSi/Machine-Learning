import numpy as np


class RidgeRegressionNE(object):

    def __init__(self):
        self._mean = None
        self._variant = None
        self._theta = None

    def _normalize(self, X):
        _X = np.array(X)
        self._mean = np.mean(_X, axis=0)
        _X = _X - self._mean
        self._variant = (np.max(_X, axis=0) - np.min(_X, axis=0)) / 2
        _X = _X / self._variant
        _X = np.append(np.ones((_X.shape[0], 1)), _X, axis=1)
        return _X

    def _re_normalize(self, X):
        _X = np.array(X)
        _X = _X - self._mean
        _X = _X / self._variant
        _X = np.append(np.ones((_X.shape[0], 1)), _X, axis=1)
        return _X

    def train(self, X, Y, la):
        _X = self._normalize(X)
        self._theta = np.linalg.pinv(_X.T.dot(_X) + la * np.eye(_X.shape[1])).dot(_X.T).dot(Y)

    def run(self, X):
        _X = self._re_normalize(X)
        return self._theta.dot(_X.T)
