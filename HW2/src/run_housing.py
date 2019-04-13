import numpy as np
from src.RidgeRegressionNE import RidgeRegressionNE
from src.LinearRegressionGD import LinearRegressionGD

def mse(Y, pred):
    n = Y.shape[0]
    return np.sum((Y.flatten() - pred.flatten()) ** 2) / n



train_data = np.genfromtxt('../../HW1/data/housing/housing_train.txt')
test_data = np.genfromtxt('../../HW1/data/housing/housing_test.txt')
X_train = train_data[:, :-1]
Y_train = train_data[:, -1]
X_test = test_data[:, :-1]
Y_test = test_data[:, -1]

print('Ridge Regression:')
rr = RidgeRegressionNE()
la = 1
print('lambda=', la)
rr.train(X_train, Y_train, la)
pred_train = rr.run(X_train)
print(' train error: ', mse(Y_train, pred_train))
pred_test = rr.run(X_test)
print(' test error: ', mse(Y_test, pred_test))

print()
print('Linear Regression (Gradient Descent):')
rr = LinearRegressionGD()
lr = 1
print('learning_rate=', lr)
rr.train(X_train, Y_train, lr)
pred_train = rr.run(X_train)
print(' train error: ', mse(Y_train, pred_train))
pred_test = rr.run(X_test)
print(' test error: ', mse(Y_test, pred_test))
