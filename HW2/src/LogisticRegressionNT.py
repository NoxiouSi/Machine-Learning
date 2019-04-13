import numpy as np


class LogisticRegressionNT(object):

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
        last_j = self._J(_X, _Y)
        while True:
            sigmoid_value = LogisticRegressionNT.sigmoimd(_X.dot(self._theta))
            diff = lr * np.linalg.pinv(sigmoid_value.T.dot(1-sigmoid_value)*(_X.T.dot(_X))).dot(_X.T.dot(_Y - sigmoid_value))
            self._theta = self._theta + diff
            j = self._J(_X, _Y)
            print(j)
            #print(np.abs(j-last_j))
            if np.abs(j-last_j) < 0.1:
                break
            last_j = j
            # lr *= 0.999

    def run(self, X):
        _X = self._re_normalize(X)
        return LogisticRegressionNT.sigmoimd(_X.dot(self._theta))

    def _J(self, X, Y):
        m = X.shape[0]
        return (-Y.T.dot(np.log(LogisticRegressionNT.sigmoimd(X.dot(self._theta)))) - (1-Y).T.dot(np.log(1-LogisticRegressionNT.sigmoimd(X.dot(self._theta)))))

    @staticmethod
    def sigmoimd(Z):
        return 1.0 / (1.0 + np.exp(-Z))
