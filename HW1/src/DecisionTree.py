import numpy as np


class DecisionTree:

    def __init__(self):
        self._root = None
        self._mean = None
        self._variant = None
        self._label_dict = None

    def _normalize(self, X, Y):
        _X = np.array(X)
        self._mean = np.mean(_X, axis=0)
        _X = _X - self._mean
        self._variant = np.max(_X, axis=0) - np.min(_X, axis=0)
        _X = _X / self._variant
        labels, _Y = np.unique(Y, return_inverse=True)
        self._label_dict = {i: lab for i, lab in enumerate(labels)}
        return _X, _Y

    def _re_normalize(self, X):
        _X = np.array(X)
        _X = _X - self._mean
        _X = _X / self._variant
        return _X

    class SplitNode:
        def __init__(self, X, Y, cnt_cutoff, gain_cutoff):
            self._feature = None
            self._threshold = None
            self._left = None
            self._right = None
            self._result = None
            self._split(X, Y, cnt_cutoff, gain_cutoff)

        def _split(self, X, Y, cnt_cutoff, gain_cutoff):
            n = X.shape[0]
            l = np.max(Y) + 1
            min_score = np.inf
            if n < cnt_cutoff:
                self._result = np.argmax(np.bincount(Y))
                return

            for d in np.arange(X.shape[1]):
                values = np.sort(np.unique(X[:, d]))
                if len(values) > 100:
                    step = len(values) / 101
                else:
                    step = 1
                for i in np.arange(1, min(101, len(values))):
                    threshold = np.mean(values[int(i * step): int(i * step) + 2])
                    sum_left = np.bincount(Y[X[:, d] < threshold])
                    sum_right = np.bincount(Y[np.logical_not(X[:, d] < threshold)])
                    n_left = np.sum(sum_left)
                    n_right = np.sum(sum_right)
                    p_left = sum_left[sum_left != 0] / n_left
                    p_right = sum_right[sum_right != 0] / n_right
                    new_score = - n_left / n * np.sum(p_left * np.log2(p_left)) - n_right / n * np.sum(
                        p_right * np.log2(p_right))
                    if min_score > new_score:
                        min_score = new_score
                        self._feature = d
                        self._threshold = threshold

            sum_all = np.bincount(Y)
            sum_all = sum_all[sum_all != 0]
            if self._feature is None:
                self._result = np.argmax(np.bincount(Y))
                return

            left = X[:, self._feature] < self._threshold
            if np.sum(left) == 0 or np.sum(left) == n or sum(
                    -sum_all / n * np.log2(sum_all / n)) - min_score < gain_cutoff:
                self._result = np.argmax(np.bincount(Y))
                return
            self._left = DecisionTree.SplitNode(X[left], Y[left], cnt_cutoff, gain_cutoff)
            self._right = DecisionTree.SplitNode(X[np.logical_not(left)], Y[np.logical_not(left)], cnt_cutoff,
                                                 gain_cutoff)

        def predict(self, X):
            n = X.shape[0]
            if self._result is not None:
                return np.ones((n, 1)) * self._result
            else:
                res = np.zeros((n, 1))
                left = X[:, self._feature] < self._threshold
                res[left] = self._left.predict(X[left])
                res[np.logical_not(left)] = self._right.predict(X[np.logical_not(left)])
                return res

    def train(self, X, Y, cnt_cutoff, gain_cutoff):
        _X, _Y = self._normalize(X, Y)
        self._root = DecisionTree.SplitNode(_X, _Y, cnt_cutoff, gain_cutoff)

    def run(self, X):
        _X = self._re_normalize(X)
        pred = self._root.predict(_X)
        return np.vectorize(lambda y: self._label_dict[y])(pred)
