import numpy as np
from sklearn.metrics import roc_auc_score


class OptimalDecisionStump:

    def __init__(self):
        self.feature = None
        self.threshold = None

    def fit(self, x, y, sample_weight=None):
        m, d = x.shape
        best_score = 0
        hit = False
        for i in range(d):
            # sort x and y
            sorted_idx = np.argsort(x[:,i])
            # convert y to 1/-1
            yy = np.sign(y[sorted_idx] - 0.5)
            # compute the accumulated score
            score = np.sum(y * sample_weight) + np.cumsum(-1 * yy * sample_weight[sorted_idx])
            # choose the threshold which produces the optimal accuracy (ignoring duplicated y value)
            for j in range(m-1):
                if x[sorted_idx[j], i] != x[sorted_idx[j+1], i] and 0.0001 < score[j] < 0.9999 and best_score < np.abs(0.5-score[j]):
                    best_score = np.abs(0.5-score[j])
                    self.feature = i
                    self.threshold = sum((x[sorted_idx[j],i], x[sorted_idx[j + 1], i]))/2
                    hit = True
        if not hit:
            raise RuntimeError('Split Fail：' + str(y.sum()) + '/' + str(y.shape[0]))

    def predict(self, X):
        return X[:, self.feature] > self.threshold


class RandomDecisionStump:

    def __init__(self):
        self.feature = None
        self.threshold = None

    def fit(self, x, y, sample_weight=None):
        m, d = x.shape
        self.feature = np.random.randint(d)
        self.threshold = np.random.choice(np.unique(x[:, self.feature]))

    def predict(self, X):
        return X[:, self.feature] > self.threshold


class AdaBoost:

    def __init__(self, T, classifier):
        self.classifiers = []
        self.alpha = []
        self.T = T
        self.classifier = classifier

    def fit(self, x, y, test_data=None, log_info=None):
        y_minus = np.sign(y-0.5)
        m = x.shape[0]
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
            clf.fit(x, y, sample_weight=D)
            pred = np.sign(clf.predict(x) - 0.5)
            epsilon = D.dot(y_minus != pred)
            if epsilon > 0.00001 and epsilon < 0.99999:
                self.alpha.append(0.5 * np.log((1 - epsilon) / epsilon))
            else:
                self.alpha.append(0)
            D *= np.exp(-self.alpha[-1] * y_minus * pred)
            D /= D.sum()
            self.classifiers.append(clf)
            if test_data is not None:
                training_hx += self.alpha[-1] * np.sign(self.classifiers[-1].predict(x) - 0.5)
                log_train_error.append(np.sum((training_hx > 0) != y) / m)
                log_round_error.append(epsilon)
                testing_hx += self.alpha[-1] * np.sign(self.classifiers[-1].predict(testX) - 0.5)
                log_test_error.append(np.sum((testing_hx > 0) != testY) / testX.shape[0])
                log_auc.append(roc_auc_score(testY, testing_hx))
                print('round', t + 1,
                      'round error:', log_round_error[-1], 'training error:', log_train_error[-1],
                      'testing error:', log_test_error[-1], 'auc:', log_auc[-1])
        if test_data is not None:
            return log_train_error, log_test_error, log_round_error, log_auc

    def predict(self, x):
        hx = np.zeros(x.shape[0])
        for i in range(len(self.classifiers)):
            hx += self.alpha[i] * np.sign(self.classifiers[i].predict(x)-0.5)
        return hx > 0

    def predict_prob(self, x):
        hx = np.zeros(x.shape[0])
        for i in range(len(self.classifiers)):
            hx += self.alpha[i] * np.sign(self.classifiers[i].predict(x)-0.5)
        return hx
