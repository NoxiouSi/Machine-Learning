import numpy as np


def mse(Y, pred):
    n = Y.shape[0]
    return np.sum((Y - pred) ** 2) / n


class RegressionTree:

    def __init__(self, depth_cutoff):
        self._root = None
        self._depth_cutoff = depth_cutoff

    class SplitNode:
        def __init__(self, X, Y, depth_cutoff, depth):
            self._feature = None
            self._threshold = None
            self._left = None
            self._right = None
            self._result = None
            self._split(X, Y, depth_cutoff, depth)

        def _split(self, X, Y, depth_cutoff, depth):
            n = X.shape[0]
            if depth > depth_cutoff or n < 2:
                self._result = np.mean(Y)
                return
            if n > 200:
                step = n / 201
            else:
                step = 1
            for d in np.arange(X.shape[1]):
                order = np.argsort(X[:, d])
                max_score = -1
                sum_left = 0
                sum_right = sum(Y)
                for i in range(1, min(201, n)):
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
            if self._threshold is None:
                raise RuntimeError("split fail")
            left = X[:, self._feature] < self._threshold
            if np.sum(left) == 0 or np.sum(left) == n:
                self._result = np.mean(Y)
                return
            self._left = RegressionTree.SplitNode(X[left], Y[left], depth_cutoff, depth + 1)
            self._right = RegressionTree.SplitNode(X[np.logical_not(left)], Y[np.logical_not(left)], depth_cutoff,
                                                   depth + 1)

        def predict(self, X):
            n = X.shape[0]
            if self._result is not None:
                return np.ones(n) * self._result
            else:
                res = np.zeros(n)
                left = X[:, self._feature] < self._threshold
                res[left] = self._left.predict(X[left])
                res[np.logical_not(left)] = self._right.predict(X[np.logical_not(left)])
                return res

    def fit(self, x, y):
        self._root = RegressionTree.SplitNode(x, y, self._depth_cutoff, 0)

    def predict(self, x):
        if not self._root:
            return None
        return self._root.predict(x)


class GradientDescentBoostingTree:

    def __init__(self, num_of_trees, classifier=lambda: RegressionTree(depth_cutoff=2)):
        self.t = num_of_trees
        self.trees = None
        self.clf = classifier

    def fit(self, x, y, test_data=None):
        y = np.array(y)
        self.trees = []
        for i in range(self.t):
            clf = self.clf()
            clf.fit(x, y)
            pred = clf.predict(x)
            self.trees.append(clf)

            log_str = 'Epoch ' + str(i) + ' Round error: ' + str(mse(y, pred))
            log_str += ' Training error: ' + str(mse(train_y, self.predict(train_x)))
            if test_data is not None:
                test_x, test_y = test_data
                log_str += ' Testing error: ' + str(mse(test_y, self.predict(test_x)))
            print(log_str)
            y = y - pred

    def predict(self, x):
        return sum(clf.predict(x) for clf in self.trees)


if __name__ == '__main__':

    _debug = False

    train_data = np.genfromtxt('../HW1/data/housing/housing_train.txt')
    test_data = np.genfromtxt('../HW1/data/housing/housing_test.txt')
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1]
    test_x = test_data[:, :-1]
    test_y = test_data[:, -1]


    def normalize(X):
        _X = np.array(X)
        _mean = np.mean(_X, axis=0)
        _X = _X - _mean
        _variant = np.max(_X, axis=0) - np.min(_X, axis=0)
        _X = _X / _variant
        return _X


    x = np.append(train_x, test_x, axis=0)
    x = normalize(x)
    train_x = x[:train_y.shape[0], :]
    test_x = x[train_y.shape[0]:, :]

    if _debug:
        gdbt = GradientDescentBoostingTree(100)
        gdbt.fit(train_x, train_y, test_data=(test_x, test_y))
        print('Testing error:', mse(test_y, gdbt.predict(test_x)))
    else:
        gdbt = GradientDescentBoostingTree(10)
        gdbt.fit(train_x, train_y)
        print('Testing error:', mse(test_y, gdbt.predict(test_x)))
