import numpy as np
from sklearn.metrics import roc_auc_score


class OptimalDecisionStump:

    def __init__(self):
        self.feature = None
        self.threshold = None

    def fit(self, X, weight, Y):
        m, d = X.shape
        best_score = 0
        hit = False
        for i in range(d):
            sorted_idx = np.argsort(X[:, i])
            acc = (Y == 1).dot(weight)
            for j in range(len(sorted_idx) - 1):
                if Y[sorted_idx[j]] == -1:
                    acc += weight[sorted_idx[j]]
                else:
                    acc -= weight[sorted_idx[j]]
                if X[sorted_idx[j], i] != X[sorted_idx[j + 1], i] and best_score < np.abs(0.5 - acc) < 0.49999:
                    best_score = np.abs(0.5 - acc)
                    self.feature = i
                    self.threshold = (X[sorted_idx[j], i] + X[sorted_idx[j + 1], i]) / 2
                    hit = True
        if not hit:
            raise RuntimeError('Split Fail：' + str(Y.sum()) + '/' + str(Y.shape[0]))

    def predict(self, X):
        return (X[:, self.feature] > self.threshold) * 2 - 1


class RandomDecisionStump:

    def __init__(self):
        self.feature = None
        self.threshold = None

    def fit(self, X, weight, Y):
        m, d = X.shape
        self.feature = np.random.randint(d)
        self.threshold = np.random.choice(np.unique(X[:, self.feature]))

    def predict(self, X):
        return (X[:, self.feature] > self.threshold) * 2 - 1


class HundredFoldDecisionStump:

    def __init__(self):
        self.feature = None
        self.threshold = None

    def fit(self, X, weight, Y):
        m, d = X.shape
        best_score = 0
        hit = False
        for i in range(d):
            sorted_idx = np.argsort(X[:, i])
            for j in range(1,101):
                threshold = X[sorted_idx[int(m/101*j)],i]
                acc = (Y[X[:,i] < threshold] == 0).sum() + (Y[X[:,i] >= threshold] == 1).sum()
                acc /= m
                if best_score < np.abs(0.5 - acc) < 0.49999:
                    best_score = np.abs(0.5 - acc)
                    self.feature = i
                    self.threshold = threshold
                    hit = True
        if not hit:
            raise RuntimeError('Split Fail：' + str(Y.sum()) + '/' + str(Y.shape[0]))

    def predict(self, X):
        return (X[:, self.feature] > self.threshold) * 2 - 1


class Booster:

    def __init__(self, T, classifier):
        self.classifiers = []
        self.alpha = []
        self.T = T
        self.classifier = classifier

    def fit(self, X, Y, test_data=None, log_info=None):
        m, d = X.shape
        D = np.ones(m) / m
        training_hx = np.zeros(m)
        if test_data is not None:
            log_auc = []
            log_train_error = []
            log_test_error = []
            log_round_error = []
            testX, testY = test_data
            testing_hx = np.zeros(testX.shape[0])
        for t in range(self.T):
            if log_info is not None:
                print(log_info, 'classifier #', t+1)
            # print(D)
            clf = self.classifier()
            clf.fit(X, D, Y)
            pred = clf.predict(X)
            epsilon = D.dot(Y != pred)
            if epsilon > 0.00001 and epsilon < 0.99999:
                self.alpha.append(0.5 * np.log((1 - epsilon) / epsilon))
            else:
                self.alpha.append(0)
            D *= np.exp(-self.alpha[-1] * Y * pred)
            D /= D.sum()
            self.classifiers.append(clf)
            if test_data is not None:
                training_hx += self.alpha[-1] * self.classifiers[-1].predict(X)
                log_train_error.append(np.sum(((training_hx > 0) * 2 - 1) != Y) / m)
                log_round_error.append(epsilon)
                testing_hx += self.alpha[-1] * self.classifiers[-1].predict(testX)
                log_test_error.append(np.sum(((testing_hx > 0) * 2 - 1) != testY) / testX.shape[0])
                log_auc.append(roc_auc_score(testY, testing_hx))
                print('round', t + 1, 'Feature', self.classifiers[-1].feature,
                      'Threshold', self.classifiers[-1].threshold,
                      'round error:', log_round_error[-1], 'training error:', log_train_error[-1],
                      'testing error:', log_test_error[-1], 'auc:', log_auc[-1])
        if test_data is not None:
            return log_train_error, log_test_error, log_round_error, log_auc

    def predict(self, X):
        hx = np.zeros(X.shape[0])
        for i in range(len(self.classifiers)):
            hx += self.alpha[i] * self.classifiers[i].predict(X)
        return np.sign(hx)

    def predict_prob(self, X):
        hx = np.zeros(X.shape[0])
        for i in range(len(self.classifiers)):
            hx += self.alpha[i] * self.classifiers[i].predict(X)
        return hx


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    clf = OptimalDecisionStump
    k = 10
    T = 100
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
        Y_train = train_data[:, -1] * 2 - 1
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1] * 2 - 1

        nb = Booster(T, clf)
        log_train_error, log_test_error, log_round_error, log_auc = nb.fit(X_train, Y_train,
                                                                           test_data=(X_test, Y_test))
        plt.figure()
        plt.plot(log_auc, 'r')
        plt.ylim(top=1.0)
        plt.title('AUC')
        plt.savefig('q1result/auc-' + clf.__name__)
        plt.close()

        plt.figure()
        line1, = plt.plot(log_train_error, 'b')
        line2, = plt.plot(log_test_error, 'r')
        plt.legend([line1, line2], ['Train Error', 'Test Error'])
        plt.title('Train/Test Error')
        plt.savefig('q1result/train_test error-' + clf.__name__)
        plt.close()

        plt.figure()
        plt.plot(log_round_error, 'r')
        plt.title('Round Error')
        plt.ylim(top=1.0, bottom=0.0)
        plt.savefig('q1result/round error-' + clf.__name__)
        plt.close()

        Y_pred = nb.predict(X_test)

        m = len(Y_pred)
        print('epoch', i + 1, 'accuracy', (Y_pred == Y_test).sum(), '/', m)
        ac += (Y_pred == Y_test).sum()

    print()
    print('Overall accuracy', ac / l)
