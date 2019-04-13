import numpy as np

data = np.genfromtxt('../HW1/data/spambase/spambase.data', delimiter=',')

X = data[:, :-1]
Y = data[:, -1]


class NB:
    def __init__(self):
        self.mu = None
        self.p_f0_y0 = None
        self.p_f1_y0 = None
        self.p_f0_y1 = None
        self.p_f1_y1 = None
        self.py0 = None
        self.py1 = None

    def train(self, X, Y):
        self.mu = X.mean(axis=0, keepdims=True)
        _X = (X > self.mu) * 1

        m, d = _X.shape

        self.p_f0_y0 = (np.array([np.sum(Y[_X[:, i] == 0] == 0) for i in range(d)]) + 0.1) / (np.sum(Y == 0) + 0.2)
        self.p_f1_y0 = (np.array([np.sum(Y[_X[:, i] == 1] == 0) for i in range(d)]) + 0.1) / (np.sum(Y == 0) + 0.2)
        self.p_f0_y1 = (np.array([np.sum(Y[_X[:, i] == 0] == 1) for i in range(d)]) + 0.1) / (np.sum(Y == 1) + 0.2)
        self.p_f1_y1 = (np.array([np.sum(Y[_X[:, i] == 1] == 1) for i in range(d)]) + 0.1) / (np.sum(Y == 1) + 0.2)

        self.py0 = np.sum(Y == 0) / m
        self.py1 = np.sum(Y == 1) / m

    def predict(self, X, raw_diff=False, threshold=0.0):
        _X = (X > self.mu) * 1
        logp0 = _X.dot(np.log(self.p_f1_y0)) + np.logical_not(_X).dot(np.log(self.p_f0_y0)) + np.log(self.py0)
        logp1 = _X.dot(np.log(self.p_f1_y1)) + np.logical_not(_X).dot(np.log(self.p_f0_y1)) + np.log(self.py1)

        if raw_diff:
            return logp1 - logp0
        else:
            return (logp1 - logp0 > threshold) * 1


if __name__ == '__main__':

    k = 10
    data = np.genfromtxt('../HW1/data/spambase/spambase.data', delimiter=',')
    l = data.shape[0]
    indexes = np.arange(l)
    np.random.shuffle(indexes)

    ac = 0

    for i in range(k):
        idx = np.arange(l)
        cv_set = indexes[int(i * l / k):int((i + 1) * l / k)]
        train_data = data[np.logical_not(np.isin(idx, cv_set)), :]
        test_data = data[cv_set, :]
        X_train = train_data[:, :-1]
        Y_train = train_data[:, -1]
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1]

        nb = NB()
        nb.train(X_train, Y_train)
        Y_pred = nb.predict(X_test, threshold=-0.5)

        m = len(Y_pred)
        tp = np.logical_and(Y_pred == 1, Y_test == 1).sum()
        fp = np.logical_and(Y_pred == 1, Y_test == 0).sum()
        fn = np.logical_and(Y_pred == 0, Y_test == 1).sum()
        tn = np.logical_and(Y_pred == 0, Y_test == 0).sum()
        print('epoch', i, 'accuracy', (Y_pred == Y_test).sum(), '/', m, 'tp', tp, 'fp', fp, 'fn', fn, 'tn', tn)
        ac += (Y_pred == Y_test).sum()

    print('accuracy', ac / l)
