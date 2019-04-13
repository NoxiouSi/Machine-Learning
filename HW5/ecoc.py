import os
from scipy.sparse import csr_matrix, lil_matrix
from boosting import *
from threading import Thread

class ECOC:
    def __init__(self, ecoc_func, k=-1):
        """
        Construct a ecoc classifier with given ecoc function and k
        :param ecoc_func: the ecoc function to be trained
        :param k: length of code, -1 means exhaustive
        """

        self.k = k
        self.coding = None
        self.y_to_code_map = None
        self.code_to_y_map = None
        self.ecoc_func = ecoc_func
        self.funcs = None

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
            f.fit(self.x, self.y, log_info='ECOC function #' + str(self.id+1))
            pred = f.predict(self.x)
            print('Training function #', self.id + 1, 'Training error:', np.not_equal(pred, self.y).sum() / pred.shape[0])
            self.func_list[self.id] = f

    def fit(self, x, y):
        y_trans = self._fit_transform_y(y)
        self.funcs = [None] * self.k
        threads = []
        for i in range(self.k):
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
        else:
            for i in range(num_labels):
                self.coding[unique_lables[i]] = [np.random.randint(2) for j in range(self.k)]
        trans_y = []
        for lab in y:
            trans_y.append(self.coding[lab])
        trans_y = np.array(trans_y)
        return trans_y


def read_data(path):
    num_labels=0
    num_data=0
    num_feature=0
    with open(os.path.join(path, 'config.txt')) as f:
        for line in f:
            if line.startswith('numClasses'):
                num_labels = int(line[11:])
            elif line.startswith('numDataPoints'):
                num_data = int(line[14:])
            elif line.startswith('numFeatures'):
                num_feature = int(line[12:])
    x = lil_matrix((num_data, num_feature))
    y = np.zeros((num_data))
    with open(os.path.join(path, 'feature_matrix.txt')) as f:
        j = 0
        for line in f:
            tokens = line.split()
            y[j] = int(tokens[0])
            for i in range(1, len(tokens)):
                col, value = tokens[i].split(':')
                x[j,int(col)] = float(value)
            j += 1
    return x.toarray(), y


if __name__ == '__main__':

    print('Reading data')
    train_x, train_y = read_data('data/8newsgroup/train.trec/')
    test_x, test_y = read_data('data/8newsgroup/test.trec/')

    print('Training')
    ecoc = ECOC(ecoc_func=lambda: Booster(2, OptimalDecisionStump), k=4)
    ecoc.fit(train_x, train_y)

    print('Testing')
    train_pred = ecoc.predict(train_x)
    print('training error:', np.not_equal(train_pred, train_y).sum()/train_y.shape[0])
    test_pred = ecoc.predict(test_x)
    print('testing error:', np.not_equal(test_pred, test_y).sum()/test_y.shape[0])
