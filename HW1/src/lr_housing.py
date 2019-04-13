import numpy as np
from src.LinearRegression import LinearRegression


def mse(Y, pred):
    n = Y.shape[0]
    return np.sum((Y - pred) ** 2) / n


train_data = np.genfromtxt('../data/housing/housing_train.txt')
test_data = np.genfromtxt('../data/housing/housing_test.txt')
X_train = train_data[:, :-1]
Y_train = train_data[:, -1]
X_test = test_data[:, :-1]
Y_test = test_data[:, -1]

lr = LinearRegression()
lr.train(X_train, Y_train)

pred_train = lr.run(X_train)
print(' train error: ', mse(Y_train, pred_train))
pred_test = lr.run(X_test)
print(' test error: ', mse(Y_test, pred_test))
