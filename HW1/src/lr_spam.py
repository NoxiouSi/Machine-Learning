import numpy as np
from src.LinearRegression import LinearRegression


def accuracy(Y, pred):
    n = Y.shape[0]
    return np.sum(Y == pred) / n


def mse(Y, pred):
    n = Y.shape[0]
    return np.sum((Y - pred) ** 2) / n


k = 5
data = np.genfromtxt('../data/spambase/spambase.data', delimiter=',')
l = data.shape[0]
indexes = np.arange(l)
np.random.shuffle(indexes)

tp = 0
tn = 0
fp = 0
fn = 0

threshold = 0.42
train_ac = []
test_ac = []
for i in range(k):
    idx = np.arange(l)
    cv_set = indexes[int(i * l / k):int((i + 1) * l / k)]
    train_data = data[np.logical_not(np.isin(idx, cv_set)), :]
    test_data = data[cv_set, :]
    X_train = train_data[:, :-1]
    Y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1]

    lr = LinearRegression()
    lr.train(X_train, Y_train)

    pred_train = (lr.run(X_train) > threshold) * 1
    train_ac.append(accuracy(Y_train.T, pred_train.flatten()))

    pred_test = (lr.run(X_test) > threshold) * 1
    test_ac.append(accuracy(Y_test.T, pred_test.flatten()))

    tp += np.logical_and(Y_test.flatten(), pred_test.flatten()).sum()
    tn += np.logical_and(np.logical_not(Y_test.flatten()), np.logical_not(pred_test.flatten())).sum()
    fp += np.logical_and(Y_test.flatten(), np.logical_not(pred_test.flatten())).sum()
    fn += np.logical_and(np.logical_not(Y_test.flatten()), pred_test.flatten()).sum()

print('threshold:', threshold)
print(' train accuracy: ', np.mean(train_ac))
print(' test accuracy: ', np.mean(test_ac))
print(tp, fp, fn, tn)
