import numpy as np

data = np.genfromtxt('../HW1/data/spambase/spambase.data', delimiter=',')

X = data[:, :-1]
Y = data[:, -1]


class NBGau:
    def __init__(self):
        self.mu0 = None
        self.var0 = None
        self.mu1 = None
        self.var1 = None
        self.py0 = None
        self.py1 = None

    def train(self, X, Y):
        m,d = X.shape

        self.mu0 = X[Y == 0].mean(axis=0)
        self.var0 = X[Y == 0].var(axis=0)
        self.var0 += np.ones(d)*0.000001
        self.mu1 = X[Y == 1].mean(axis=0)
        self.var1 = X[Y == 1].var(axis=0)
        self.var1 += np.ones(d)*0.000001

        self.py0 = np.sum(Y == 0) / m
        self.py1 = np.sum(Y == 1) / m

    def predict(self, X, raw_diff=False, threshold=0.0):
        m,d = X.shape

        # logp0 = np.array([np.sum((X[i,:] - self.mu0)**2 / self.var0 * np.log(self.var0)) for i in range(m)])
        # logp1 = np.array([np.sum((X[i,:] - self.mu1)**2 / self.var1 * np.log(self.var1)) for i in range(m)])
        logp0 = np.sum(-1 * ((X - self.mu0) ** 2 / self.var0) - 1 / 2 * (np.log(2*np.pi*self.var0)), axis=1) + np.log(self.py0)
        logp1 = np.sum(-1 * ((X - self.mu1) ** 2 / self.var1) - 1 / 2 * (np.log(2*np.pi*self.var1)), axis=1) + np.log(self.py1)

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

    for t in np.arange(-10, 50.0, 1):
        print(t)
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

            nb = NBGau()
            nb.train(X_train, Y_train)
            Y_pred = nb.predict(X_test, threshold=t)

            m = len(Y_pred)

            tp = np.logical_and(Y_pred == 1, Y_test == 1).sum()
            fp = np.logical_and(Y_pred == 1, Y_test == 0).sum()
            fn = np.logical_and(Y_pred == 0, Y_test == 1).sum()
            tn = np.logical_and(Y_pred == 0, Y_test == 0).sum()
            # print('epoch', i, 'accuracy', (Y_pred == Y_test).sum(), '/', m, 'tp', tp, 'fp', fp, 'fn', fn, 'tn', tn)
            ac += (Y_pred == Y_test).sum()

        print('accuracy', ac / l)
