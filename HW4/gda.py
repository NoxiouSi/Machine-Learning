import numpy as np
from scipy.stats import multivariate_normal


class GDA():
    """
    Gaussian Discriminant Analysis
    """

    def __init__(self):
        self.mu0 = None
        self.mu1 = None
        self.sigma = None

    def train(self, x, y):
        """
        """
        self.mu0 = np.mean(x[y == 0], axis=0, keepdims=True)
        self.mu1 = np.mean(x[y == 1], axis=0, keepdims=True)

        n_x = x[y == 0] - self.mu0
        p_x = x[y == 1] - self.mu1

        self.sigma = ((n_x.T).dot(n_x) + (p_x.T).dot(p_x)) / x.shape[0]
        self.sigma += np.eye(self.sigma.shape[0]) * 0.00001

    def predict(self, x):
        m=x.shape[0]
        result = np.zeros((m,1))
        for i in range(m):
            n_x = x - self.mu0
            p_x = x - self.mu1
            sigma_inv = np.linalg.pinv(self.sigma)

            py0 = -n_x[i,:].dot(sigma_inv).dot(n_x[i,:].T)
            py1 = -p_x[i,:].dot(sigma_inv).dot(p_x[i,:].T)

            result[i,0] = (py1 > py0) * 1
        return result


if __name__ == '__main__':

    k = 10
    data = np.genfromtxt('../HW1/data/spambase/spambase.data', delimiter=',')
    l = data.shape[0]
    indexes = np.arange(l)
    np.random.shuffle(indexes)

    ac = []

    for i in range(k):
        idx = np.arange(l)
        cv_set = indexes[int(i * l / k):int((i + 1) * l / k)]
        train_data = data[np.logical_not(np.isin(idx, cv_set)), :]
        test_data = data[cv_set, :]
        X_train = train_data[:, :-1]
        Y_train = train_data[:, -1]
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1]

        gda = GDA()
        gda.train(X_train, Y_train)
        Y_pred = gda.predict(X_test)

        m=len(Y_pred)

        print('epoch', i, 'accuracy', (Y_pred == Y_test.reshape(m,1)).sum(), '/', m)
        ac.append((Y_pred == Y_test.reshape(m,1)).sum() / m)
    print('accuracy', np.average(ac))
