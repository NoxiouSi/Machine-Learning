import numpy as np


class RegressionTree:

    def __init__(self):
        self._root = None
        self._mean = None
        self._variant = None

    def _normalize(self, X):
        _X = np.array(X)
        self._mean = np.mean(_X, axis=0)
        _X = _X - self._mean
        self._variant = np.max(_X, axis=0) - np.min(_X, axis=0)
        _X = _X / self._variant
        return _X

    def _re_normalize(self, X):
        _X = np.array(X)
        _X = _X - self._mean
        _X = _X / self._variant
        return _X

    class SplitNode:
        def __init__(self, X, Y, cnt_cutoff, var_cutoff):
            self._feature = None
            self._threshold = None
            self._left = None
            self._right = None
            self._result = None
            self._split(X, Y, cnt_cutoff, var_cutoff)

        def _split(self, X, Y, cnt_cutoff, var_cutoff):
            n = X.shape[0]
            if n < cnt_cutoff or np.var(Y) < var_cutoff:
                self._result = np.mean(Y)
                return
            if n > 100:
                step = n / 101
            else:
                step = 1
            for d in np.arange(X.shape[1]):
                order = np.argsort(X[:, d])
                max_score = 0
                sum_left = 0
                sum_right = sum(Y)
                for i in range(1, min(101, n)):
                    threshold = np.mean(X[order[int(i * step): int(i * step) + 2], d])
                    delta = np.sum(Y[order[int((i - 1) * step):int(i * step)]])
                    sum_left += delta
                    sum_right -= delta
                    n_left = int(i * step)
                    n_right = n - n_left
                    new_score = sum_left ** 2 / n_left + sum_right ** 2 / n_right
                    if max_score < new_score:
                        max_score = new_score
                        self._feature = d
                        self._threshold = threshold
            left = X[:, self._feature] < self._threshold
            if np.sum(left) == 0 or np.sum(left) == n:
                self._result = np.mean(Y)
                return
            self._left = RegressionTree.SplitNode(X[left], Y[left], cnt_cutoff, var_cutoff)
            self._right = RegressionTree.SplitNode(X[np.logical_not(left)], Y[np.logical_not(left)], cnt_cutoff,
                                                   var_cutoff)

        def predict(self, X):
            n = X.shape[0]
            if self._result:
                return np.ones((n, 1)) * self._result
            else:
                res = np.zeros((n, 1))
                left = X[:, self._feature] < self._threshold
                res[left] = self._left.predict(X[left])
                res[np.logical_not(left)] = self._right.predict(X[np.logical_not(left)])
                return res

    def train(self, X, Y, cnt_cutoff, var_cutoff):
        _X = self._normalize(X)
        self._root = RegressionTree.SplitNode(_X, Y, cnt_cutoff, var_cutoff)

    def run(self, X):
        if not self._root:
            return None
        _X = self._re_normalize(X)
        return self._root.predict(_X)
