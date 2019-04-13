import numpy as np
from sklearn.linear_model import Lasso

class LogisticRegressionNT(object):

    def __init__(self, learning_rate=0.1, regularizer=None, lamb=0.1, tol=0.0001):
        self._mean = None
        self._variant = None
        self._theta = None
        self.learning_rate = learning_rate
        if regularizer == 'LASSO' or regularizer == 'RIDGE' or regularizer is None:
            self.regularizer = regularizer
        else:
            raise ValueError('Invalid regularizer')
        self.lamb = lamb
        self.tol = tol

    def fit(self, x, y, test_data=None):
        self._theta = np.random.rand(x.shape[1]) / x.shape[1]
        last_j = self.loss_function(x, y)
        while True:
            der = self.derivative_function(x, y)
            self._theta -= self.learning_rate * der
            j = self.loss_function(x, y)
            if test_data is not None:
                test_x, test_y = test_data
                pred = nb.predict(train_x)
                print('Loss', j, end=' ')
                print("Train error:", np.mean(np.not_equal(pred, train_y)), end=' ')
                pred = nb.predict(test_x)
                print("Test error:", np.mean(np.not_equal(pred, test_y)))

            # print(np.abs(j-last_j))
            if np.abs(j - last_j) < self.tol:
                break
            last_j = j
            # lr *= 0.999

    def predict(self, x):
        return self.sigmoid(x.dot(self._theta)) > 0.5

    def loss_function(self, x, y):
        a = np.minimum(np.maximum(self.sigmoid(x.dot(self._theta)), 0.0001), 0.9999)
        loss = -(y.dot(np.log(a)) + (1 - y).dot(np.log(1 - a)))
        if self.regularizer == 'LASSO':
            loss += self.lamb * self.lasso(self._theta)
        elif self.regularizer == 'RIDGE':
            loss += self.lamb * self.ridge(self._theta)
        return loss

    def derivative_function(self, x, true_label):
        # H(X) = sigma_i[h(x_i) * (1-h(x_i)) * x_i * x_i.T]
        hx = self.sigmoid(x.dot(self._theta))
        hessian = (x.T * hx * (1 - hx)).dot(x)
        if self.regularizer == 'RIDGE':
            hessian += 2 * np.eye(hessian.shape[0]) * self.lamb
        # L'(x) = sigma_i[(h(x_i) - y_i) * x_i]
        dl = x.T.dot(hx - true_label)
        if self.regularizer == 'LASSO':
            dl += self.lamb * np.sign(self._theta)
        elif self.regularizer == 'RIDGE':
            dl += 2 * self.lamb * self._theta
        # H(x)^(-1) * L'(x)
        deriv = np.linalg.pinv(hessian).dot(dl)
        # print(deriv)
        return deriv

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def lasso(w):
        return np.sum(np.abs(w))

    @staticmethod
    def ridge(w):
        return np.dot(w, w)


def normalize(x):
    x = np.array(x)
    x = x - np.mean(x, axis=0)
    x /= np.var(x, axis=0)
    return x


if __name__ == '__main__':
    #
    # print('Original Dataset')
    # print('Reading data')
    # data = np.genfromtxt('../HW1/data/spambase/spambase.data', delimiter=',')
    # train_x = data[:, :-1]
    # train_y = data[:, -1]
    # test_x = train_x
    # test_y = train_y


    # print('No regularization: Training model')
    # nb = LogisticRegressionNT()
    # nb.fit(train_x, train_y)
    #
    # pred = nb.predict(test_x)
    # print("Accuracy:", np.mean(np.equal(pred, test_y)))
    #
    # print('LASSO regularization: Training model')
    # nb = LogisticRegressionNT(regularizer='LASSO')
    # nb.fit(train_x, train_y)
    #
    # pred = nb.predict(test_x)
    # print("Accuracy:", np.mean(np.equal(pred, test_y)))
    #
    # print('RIDGE regularization: Training model')
    # nb = LogisticRegressionNT(regularizer='RIDGE')
    # nb.fit(train_x, train_y)
    #
    # pred = nb.predict(test_x)
    # print("Accuracy:", np.mean(np.equal(pred, test_y)))


    print('Reading data')
    train_x = np.genfromtxt('spam_polluted/train_feature.txt', delimiter=' ')
    train_y = np.genfromtxt('spam_polluted/train_label.txt', delimiter=' ')

    test_x = np.genfromtxt('spam_polluted/test_feature.txt', delimiter=' ')
    test_y = np.genfromtxt('spam_polluted/test_label.txt', delimiter=' ')

    # x_norm = normalize(np.append(train_x, test_x, axis=0))
    # train_x = x_norm[:train_x.shape[0], :]
    # test_x = x_norm[train_x.shape[0]:, :]

    # train_x = np.append(np.ones((train_x.shape[0], 1)), train_x, axis=1)
    # test_x = np.append(np.ones((test_x.shape[0], 1)), test_x, axis=1)

    #
    # train_x = np.array([[1,0],[1,1],[2,0],[2,1],[3,0],[3,1],[5,3],[5,4],[6,3],[6,4],[7,3],[7,4]])
    # train_x = np.append(np.ones((train_x.shape[0], 1)), train_x, axis=1)
    # train_y = np.array([1,1,1,1,1,1,0,0,0,0,0,0])
    # test_x = train_x
    # test_y = train_y

    # print('No regularization: Training model')
    # nb = LogisticRegressionNT(learning_rate=0.1, tol=20)
    # nb.fit(train_x, train_y, test_data=(test_x, test_y))
    #
    # pred = nb.predict(test_x)
    # print("Accuracy:", np.mean(np.equal(pred, test_y)))

    print('LASSO regularization: Training model')
    nb = LogisticRegressionNT(regularizer='LASSO', learning_rate=0.1, lamb=1, tol=0.1)
    nb.fit(train_x, train_y, test_data=(test_x, test_y))

    pred = nb.predict(test_x)
    print("Accuracy:", np.mean(np.equal(pred, test_y)))

    # print('RIDGE regularization: Training model')
    # nb = LogisticRegressionNT(regularizer='RIDGE', learning_rate=0.1, lamb=10)
    # nb.fit(train_x, train_y, test_data=(test_x, test_y))
    #
    # pred = nb.predict(test_x)
    # print("Accuracy:", np.mean(np.equal(pred, test_y)))

    from sklearn.linear_model import LogisticRegression

    # nb = LogisticRegression(solver='newton-cg')
    # nb.fit(train_x, train_y)
    #
    # pred = nb.predict(test_x)
    # print("Accuracy:", np.mean(np.equal(pred, test_y)))
