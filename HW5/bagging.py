import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import mode


def error_rate(Y, pred):
    n = Y.shape[0]
    return np.sum(np.not_equal(Y, pred)) / n


class DecisionTree:

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
            l = np.max(Y) + 1
            min_score = np.inf
            if depth > depth_cutoff or n < 2:
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
                    temp = Y[X[:, d] < threshold]
                    sum_left = np.bincount(temp)
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

            if self._feature is None:
                self._result = np.argmax(np.bincount(Y))
                return

            left = X[:, self._feature] < self._threshold
            if np.sum(left) == 0 or np.sum(left) == n:
                self._result = np.argmax(np.bincount(Y))
                return
            self._left = DecisionTree.SplitNode(X[left], Y[left], depth_cutoff, depth+1)
            self._right = DecisionTree.SplitNode(X[np.logical_not(left)], Y[np.logical_not(left)], depth_cutoff, depth+1)

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
        self._root = DecisionTree.SplitNode(x, y, self._depth_cutoff, 0)

    def predict(self, x):
        pred = self._root.predict(x)
        return pred


class Bagging:
    def __init__(self, num_of_trees, classifier=lambda: DecisionTree(depth_cutoff=2)):
        self.t = num_of_trees
        self.trees = None
        self.clf = classifier

    def fit(self, x, y, test_data=None):
        self.trees = []
        n = x.shape[0]
        indexes = np.arange(n)
        for i in range(self.t):
            samples = np.random.choice(indexes, int(0.3*n))
            train_x = x[samples, :]
            train_y = y[samples]
            clf = self.clf()
            clf.fit(train_x, train_y)
            self.trees.append(clf)

            pred = clf.predict(train_x)
            log_str = 'Epoch ' + str(i) + ' Round error: ' + str(error_rate(train_y, pred))
            log_str += ' Training error: ' + str(error_rate(y, self.predict(x)))
            if test_data is not None:
                test_x, test_y = test_data
                log_str += ' Testing error: ' + str(error_rate(test_y, self.predict(test_x)))
            print(log_str)

    def predict(self, x):
        res = np.zeros((x.shape[0], len(self.trees)))
        for i,clf in enumerate(self.trees):
            res[:, i] = clf.predict(x)
        mod, count = mode(res, axis=1)
        return mod.flatten()

def normalize(X, Y):
    _X = np.array(X)
    _mean = np.mean(_X, axis=0)
    _X = _X - _mean
    _variant = np.max(_X, axis=0) - np.min(_X, axis=0)
    _X = _X / _variant
    labels, _Y = np.unique(Y, return_inverse=True)
    return _X, _Y


if __name__ == '__main__':
    data = np.genfromtxt('../HW1/data/spambase/spambase.data', delimiter=',')
    x = data[:, :-1]
    y = data[:, -1]
    x, y = normalize(x, y)
    kf = KFold(n_splits=5, shuffle=True)
    i = 0
    train_error = 0
    test_error = 0
    for training_set, test_set in kf.split(data):
        train_x = x[training_set, :]
        train_y = y[training_set]
        test_x = x[test_set, :]
        test_y = y[test_set]

        bag = Bagging(50)
        bag.fit(train_x, train_y, test_data=(test_x, test_y))
        pred = bag.predict(train_x)
        train_e = np.not_equal(pred, train_y).sum()

        pred = bag.predict(test_x)
        test_e = np.not_equal(pred, test_y).sum()
        print('epoch', i, 'training error', train_e, 'test error', test_e)
        train_error += train_e
        test_error += test_e

        i += 1

    m = data.shape[0]
    print('training error:', train_error / m, 'test error:', test_error / m)

