import numpy as np

data = np.genfromtxt('../HW1/data/spambase/spambase.data', delimiter=',')

X = data[:, :-1]
Y = data[:, -1]


class NB4Bin:
    def __init__(self):
        self.bin = 4
        self.thresholds = None
        self.p_f_y0 = None
        self.p_f_y1 = None
        self.py0 = None
        self.py1 = None

    def train(self, X, Y):
        mu = X.mean(axis=0)
        mu_y0 = X[Y == 0].mean(axis=0)
        mu_y1 = X[Y == 1].mean(axis=0)

        self.thresholds = np.array([mu,mu_y0,mu_y1])
        self.thresholds.sort(axis=0)

        xs = []
        xs.append(X < self.thresholds[0])
        for b in range(self.bin-2):
            xs.append(np.logical_and(X >= self.thresholds[b], X < self.thresholds[b+1]))
        xs.append(X > self.thresholds[self.bin-2])

        m, d = X.shape

        self.p_f_y0 = []
        for b in range(self.bin):
            self.p_f_y0.append(
                (np.array([np.sum(Y[xs[b][:,i]] == 0) for i in range(d)]) + 0.1) / (np.sum(Y == 0) + 0.1*self.bin))

        self.p_f_y1 = []
        for b in range(self.bin):
            self.p_f_y1.append(
                (np.array([np.sum(Y[xs[b][:,i]] == 1) for i in range(d)]) + 0.1) / (np.sum(Y == 1) + 0.1*self.bin))

        self.py0 = np.sum(Y == 0) / m
        self.py1 = np.sum(Y == 1) / m

    def predict(self, X, raw_diff=False, threshold=0.0):
        xs = []
        xs.append(X < self.thresholds[0])
        for b in range(self.bin - 2):
            xs.append(np.logical_and(X >= self.thresholds[b], X < self.thresholds[b + 1]))
        xs.append(X > self.thresholds[self.bin - 2])

        logp0 = np.log(self.py0)
        logp1 = np.log(self.py1)
        for b in range(self.bin):
            logp0 += xs[b].dot(np.log(self.p_f_y0[b]))
            logp1 += xs[b].dot(np.log(self.p_f_y1[b]))

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

        nb = NB4Bin()
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
