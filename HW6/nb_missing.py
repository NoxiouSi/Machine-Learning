import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

class NaiveBayesClassifier:
    def __init__(self):
        self.mu = None
        self.p_f0_y0 = None
        self.p_f1_y0 = None
        self.p_f0_y1 = None
        self.p_f1_y1 = None
        self.py0 = None
        self.py1 = None

    def fit(self, x, y):
        self.mu = np.nanmean(x, axis=0, keepdims=True)
        x = np.sign(x - self.mu)

        c_f0_y0 = np.equal(-1*x.T, 1-y).T.sum(axis=0) + 0.1
        c_f1_y0 = np.equal(x.T, 1-y).T.sum(axis=0) + 0.1
        c_f0_y1 = np.equal(-1*x.T, y).T.sum(axis=0) + 0.1
        c_f1_y1 = np.equal(x.T, y).T.sum(axis=0) + 0.1

        self.p_f0_y0 = c_f0_y0 / (c_f0_y0 + c_f1_y0)
        self.p_f1_y0 = c_f1_y0 / (c_f0_y0 + c_f1_y0)
        self.p_f0_y1 = c_f0_y1 / (c_f0_y1 + c_f1_y1)
        self.p_f1_y1 = c_f1_y1 / (c_f0_y1 + c_f1_y1)

        self.py0 = np.equal(y, 0).mean()
        self.py1 = np.equal(y, 1).mean()

    def predict(self, x, threshold=0.0):
        x = np.sign(x - self.mu)

        logp0 = np.equal(x, 1).dot(np.log(self.p_f1_y0)) + \
                np.equal(x, -1).dot(np.log(self.p_f0_y0)) + \
                np.log(self.py0)
        logp1 = np.equal(x, 1).dot(np.log(self.p_f1_y1)) + \
                np.equal(x, -1).dot(np.log(self.p_f0_y1)) + \
                np.log(self.py1)

        return (logp1 - logp0 > threshold) * 1

    def predict_proba(self, x):
        x = np.sign(x - self.mu)

        logp0 = np.equal(x, 1).dot(np.log(self.p_f1_y0)) + \
                np.equal(x, -1).dot(np.log(self.p_f0_y0)) + \
                np.log(self.py0)
        logp1 = np.equal(x, 1).dot(np.log(self.p_f1_y1)) + \
                np.equal(x, -1).dot(np.log(self.p_f0_y1)) + \
                np.log(self.py1)


        return logp1 - logp0


if __name__ == '__main__':
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', r'invalid value encountered in sign')

        train_data = pd.read_csv('spam_missing/20_percent_missing_train.txt', na_values='nan', header=None).values
        test_data = pd.read_csv('spam_missing/20_percent_missing_test.txt', na_values='nan', header=None).values

        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]

        nb = NaiveBayesClassifier()
        nb.fit(train_x, train_y)

        pred = nb.predict(train_x)
        print('Train Accuracy:', np.mean(np.equal(pred, train_y)))

        pred = nb.predict(test_x)
        print('Test Accuracy:', np.mean(np.equal(pred, test_y)))
        prob = nb.predict_proba(test_x)
        print('auc:', roc_auc_score(test_y, prob))
