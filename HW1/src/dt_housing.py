import numpy as np
from src.RegressionTree import RegressionTree


def mse(Y, pred):
    n = Y.shape[0]
    return np.sum((Y - pred) ** 2) / n


train_data = np.genfromtxt('../data/housing/housing_train.txt')
test_data = np.genfromtxt('../data/housing/housing_test.txt')
X_train = train_data[:, :-1]
Y_train = train_data[:, -1]
X_test = test_data[:, :-1]
Y_test = test_data[:, -1]

for cnt_cutoff in np.arange(1, 20, 3):
    for var_cutoff in np.arange(5, 50, 5):
        tree = RegressionTree()
        tree.train(X_train, Y_train, cnt_cutoff, var_cutoff)

        print('min leaf count:', cnt_cutoff, 'min variant:', var_cutoff)
        pred_train = tree.run(X_train)
        print(' train error: ', mse(Y_train.T, pred_train.flatten()))

        pred_test = tree.run(X_test)
        print(' test error: ', mse(Y_test.T, pred_test.flatten()))

        print()
