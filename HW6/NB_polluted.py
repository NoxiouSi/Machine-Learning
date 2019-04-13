import numpy as np
from sklearn.decomposition import PCA

class NaiveBayesGau:
    def __init__(self):
        self.mu0 = None
        self.var0 = None
        self.mu1 = None
        self.var1 = None
        self.py0 = None
        self.py1 = None

    def fit(self, X, Y):
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

    apply_pca = True

    print('Reading data')
    train_x = np.genfromtxt('spam_polluted/train_feature.txt', delimiter=' ')
    train_y = np.genfromtxt('spam_polluted/train_label.txt', delimiter=' ')

    if apply_pca:
        pca = PCA(n_components=100)
        train_x = pca.fit_transform(train_x)
        print(pca.explained_variance_ratio_)

    print('Training model')
    nb = NaiveBayesGau()
    nb.fit(train_x, train_y)

    test_x = np.genfromtxt('spam_polluted/test_feature.txt', delimiter=' ')
    test_y = np.genfromtxt('spam_polluted/test_label.txt', delimiter=' ')
    if apply_pca:
        test_x = pca.transform(test_x)
    pred = nb.predict(test_x)
    print("Accuracy:", np.mean(np.equal(pred, test_y)))
