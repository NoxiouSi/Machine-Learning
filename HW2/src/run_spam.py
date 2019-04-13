import numpy as np
import matplotlib.pyplot as plot
from src.LogisticRegressionGD import LogisticRegressionGD
from src.LinearRegressionGD import LinearRegressionGD
from src.LogisticRegressionNT import LogisticRegressionNT
from src.RidgeRegressionNE import RidgeRegressionNE


def accuracy(Y, pred):
    n = Y.shape[0]
    return np.sum(Y == pred) / n


def mse(Y, pred):
    n = Y.shape[0]
    return np.sum((Y - pred) ** 2) / n


k = 5
data = np.genfromtxt('../../HW1/data/spambase/spambase.data', delimiter=',')
l = data.shape[0]
indexes = np.arange(l)
np.random.shuffle(indexes)

print('Ridge Regression:')
la = 1
threshold = 0.42
train_ac = []
test_ac = []
# tp = 0
# tn = 0
# fp = 0
# fn = 0
for i in range(k):
    idx = np.arange(l)
    cv_set = indexes[int(i * l / k):int((i + 1) * l / k)]
    train_data = data[np.logical_not(np.isin(idx, cv_set)), :]
    test_data = data[cv_set, :]
    X_train = train_data[:, :-1]
    Y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1]

    lr = RidgeRegressionNE()
    lr.train(X_train, Y_train, la)

    pred_train = (lr.run(X_train) > threshold) * 1
    train_ac.append(accuracy(Y_train.T, pred_train.flatten()))

    pred_test = (lr.run(X_test) > threshold) * 1
    test_ac.append(accuracy(Y_test.T, pred_test.flatten()))

    # tp += np.logical_and(Y_test.flatten(), pred_test.flatten()).sum()
    # tn += np.logical_and(np.logical_not(Y_test.flatten()), np.logical_not(pred_test.flatten())).sum()
    # fp += np.logical_and(Y_test.flatten(), np.logical_not(pred_test.flatten())).sum()
    # fn += np.logical_and(np.logical_not(Y_test.flatten()), pred_test.flatten()).sum()

print('threshold:', threshold, 'lambda', la)
print(' train accuracy: ', np.mean(train_ac))
print(' test accuracy: ', np.mean(test_ac))
# print(tp, fp, fn, tn)

print()
print('Linear Regression (Gradient Descent):')
lr = 0.1
threshold = 0.42
train_ac = []
test_ac = []
# tp = 0
# tn = 0
# fp = 0
# fn = 0
pred_linear = np.array([])
Y_linear = np.array([])
for i in range(k):
    idx = np.arange(l)
    cv_set = indexes[int(i * l / k):int((i + 1) * l / k)]
    train_data = data[np.logical_not(np.isin(idx, cv_set)), :]
    test_data = data[cv_set, :]
    X_train = train_data[:, :-1]
    Y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1]

    rr = LinearRegressionGD()
    rr.train(X_train, Y_train, lr)
    pred_train = (rr.run(X_train) > threshold) * 1
    pred_test = (rr.run(X_test) > threshold) * 1

    pred_linear = np.append(pred_linear, rr.run(X_test).flatten())
    Y_linear = np.append(Y_linear, Y_test.flatten())

    train_ac.append(accuracy(Y_train.T, pred_train.flatten()))
    test_ac.append(accuracy(Y_test.T, pred_test.flatten()))

    # tp += np.logical_and(Y_test.flatten(), pred_test.flatten()).sum()
    # tn += np.logical_and(np.logical_not(Y_test.flatten()), np.logical_not(pred_test.flatten())).sum()
    # fp += np.logical_and(Y_test.flatten(), np.logical_not(pred_test.flatten())).sum()
    # fn += np.logical_and(np.logical_not(Y_test.flatten()), pred_test.flatten()).sum()

print('threshold:', threshold, 'learning rate', lr)
print(' train accuracy: ', np.mean(train_ac))
print(' test accuracy: ', np.mean(test_ac))
# print(tp, fp, fn, tn)

print()
print('Logistic Regression (Gradient Descent):')
lr = 0.1
threshold = 0.4
train_ac = []
test_ac = []
# tp = 0
# tn = 0
# fp = 0
# fn = 0
pred_logistic = np.array([])
Y_logistic = np.array([])
for i in range(k):
    idx = np.arange(l)
    cv_set = indexes[int(i * l / k):int((i + 1) * l / k)]
    train_data = data[np.logical_not(np.isin(idx, cv_set)), :]
    test_data = data[cv_set, :]
    X_train = train_data[:, :-1]
    Y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1]

    lg = LogisticRegressionGD()
    lg.train(X_train, Y_train, lr)
    pred_train = (lg.run(X_train) > threshold) * 1
    pred_test = (lg.run(X_test) > threshold) * 1

    pred_logistic = np.append(pred_logistic, lg.run(X_test).flatten())
    Y_logistic = np.append(Y_logistic, Y_test.flatten())

    train_ac.append(accuracy(Y_train.flatten(), pred_train.flatten()))
    test_ac.append(accuracy(Y_test.flatten(), pred_test.flatten()))
    # tp += np.logical_and(Y_test.flatten(), pred_test.flatten()).sum()
    # tn += np.logical_and(np.logical_not(Y_test.flatten()), np.logical_not(pred_test.flatten())).sum()
    # fp += np.logical_and(Y_test.flatten(), np.logical_not(pred_test.flatten())).sum()
    # fn += np.logical_and(np.logical_not(Y_test.flatten()), pred_test.flatten()).sum()

print('threshold:', threshold, 'learning rate', lr)
print(' train accuracy: ', np.mean(train_ac))
print(' test accuracy: ', np.mean(test_ac))
# print(tp, fp, fn, tn)

fpr_linear = np.array([])
tpr_linear = np.array([])
fpr_logistic = np.array([])
tpr_logistic = np.array([])

for threshold in np.arange(0.0, 1.01, 0.01):
    linear_res = pred_linear > threshold
    fpr_linear = np.append(fpr_linear, np.logical_and(linear_res, np.logical_not(Y_linear)).sum() / np.logical_not(Y_linear).sum())
    tpr_linear = np.append(tpr_linear, np.logical_and(linear_res, Y_linear).sum() / Y_linear.sum())
    logistic_res = pred_logistic > threshold
    fpr_logistic = np.append(fpr_logistic, np.logical_and(logistic_res, np.logical_not(Y_logistic)).sum() / np.logical_not(Y_logistic).sum())
    tpr_logistic = np.append(tpr_logistic, np.logical_and(logistic_res, Y_logistic).sum() / Y_logistic.sum())

fpr_linear = np.insert(fpr_linear, 0, 1)
fpr_linear = np.append(fpr_linear, 0)
tpr_linear = np.insert(tpr_linear, 0, 1)
tpr_linear = np.append(tpr_linear, 0)
fpr_logistic = np.insert(fpr_logistic, 0, 1)
fpr_logistic = np.append(fpr_logistic, 0)
tpr_logistic = np.insert(tpr_logistic, 0, 1)
tpr_logistic = np.append(tpr_logistic, 0)

fig = plot.figure()
l1, = plot.plot(fpr_linear, tpr_linear, color='r', label='linear regression')
l2, = plot.plot(fpr_logistic, tpr_logistic, color='b', label='logistic regression')
plot.legend(handles=[l1, l2])
fig.show()
fig.savefig('ROC curve')

height = (tpr_linear[:-1] + tpr_linear[1:]) / 2
width = -np.diff(fpr_linear)
print('AUC(Linear):', np.sum(height*width))

height = (tpr_logistic[:-1] + tpr_logistic[1:]) / 2
width = -np.diff(fpr_logistic)
print('AUC(Logistic):', np.sum(height*width))


print()
print("Logistic Regression (Newton's method):")
lr = 0.1
threshold = 0.4
train_ac = []
test_ac = []
# tp = 0
# tn = 0
# fp = 0
# fn = 0
for i in range(k):
    idx = np.arange(l)
    cv_set = indexes[int(i * l / k):int((i + 1) * l / k)]
    train_data = data[np.logical_not(np.isin(idx, cv_set)), :]
    test_data = data[cv_set, :]
    X_train = train_data[:, :-1]
    Y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1]

    lg = LogisticRegressionNT()
    lg.train(X_train, Y_train, lr)
    pred_train = (lg.run(X_train) > threshold) * 1
    pred_test = (lg.run(X_test) > threshold) * 1

    train_ac.append(accuracy(Y_train.flatten(), pred_train.flatten()))
    test_ac.append(accuracy(Y_test.flatten(), pred_test.flatten()))
    # tp += np.logical_and(Y_test.flatten(), pred_test.flatten()).sum()
    # tn += np.logical_and(np.logical_not(Y_test.flatten()), np.logical_not(pred_test.flatten())).sum()
    # fp += np.logical_and(Y_test.flatten(), np.logical_not(pred_test.flatten())).sum()
    # fn += np.logical_and(np.logical_not(Y_test.flatten()), pred_test.flatten()).sum()

print('threshold:', threshold, 'learning rate', lr)
print(' train accuracy: ', np.mean(train_ac))
print(' test accuracy: ', np.mean(test_ac))
# print(tp, fp, fn, tn)
