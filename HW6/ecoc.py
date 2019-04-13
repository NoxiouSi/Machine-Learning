import os
from scipy.sparse import csr_matrix, lil_matrix
from boosting import *
from threading import Thread


class ECOC:
    def __init__(self, ecoc_func, k=-1, n_jobs=4):
        """
        Construct a ecoc classifier with given ecoc function and k
        :param ecoc_func: the ecoc function to be trained
        :param k: length of code, -1 means exhaustive
        """

        self.k = k
        self.coding = None
        self.ecoc_func = ecoc_func
        self.funcs = None
        self.n_jobs = n_jobs

    class TrainThread(Thread):
        def __init__(self, x, y, func, id, func_list):
            super().__init__()
            self.x = x
            self.y = y
            self.id = id
            self.func = func
            self.func_list = func_list

        def run(self):
            f = self.func()
            f.fit(self.x, self.y)
            pred = f.predict(self.x)
            print('Training function #', self.id + 1, 'Training error:', np.not_equal(pred, self.y).mean())
            self.func_list[self.id] = f

    def fit(self, x, y):
        if self.k == -1:
            k = np.power(2, len(np.unique(y)))
        elif self.k == -2:
            k = len(np.unique(y))
        else:
            k = self.k
        y_trans = self._fit_transform_y(y)
        self.funcs = [None] * k
        threads = []
        for i in range(k):
            t = ECOC.TrainThread(x, y_trans[:, i], self.ecoc_func, i, self.funcs)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def predict(self, x):
        pred = np.zeros((x.shape[0], 0))
        for f in self.funcs:
            pred = np.append(pred, f.predict(x).reshape(-1, 1), axis=1)

        labels, target = zip(*self.coding.items())
        res = []
        for vec in pred:
            res.append(labels[self._smallest_hamming(target, vec)])
        return np.array(res)

    def _smallest_hamming(self, target, vec):
        dist = np.logical_xor(target, vec).sum(axis=1)
        return np.argmin(dist)

    def _fit_transform_y(self, y):
        unique_lables = np.unique(y)
        num_labels = len(unique_lables)
        self.coding = {}
        if self.k == -1:
            self.k = np.power(2, num_labels - 1)
            self.coding[unique_lables[0]] = [1] * self.k
            for i in range(0, num_labels - 1):
                self.coding[unique_lables[i + 1]] = \
                    ([0] * np.power(2, num_labels - i - 2) + [1] * np.power(2, num_labels - i - 2)) * np.power(2, i)
        elif self.k == -2:
            self.coding = {label: np.eye(num_labels)[i, :] for i, label in enumerate(unique_lables)}
        else:
            for i in range(num_labels):
                self.coding[unique_lables[i]] = [np.random.randint(2) for j in range(self.k)]
        trans_y = []
        for lab in y:
            trans_y.append(self.coding[lab])
        trans_y = np.array(trans_y)
        return trans_y
