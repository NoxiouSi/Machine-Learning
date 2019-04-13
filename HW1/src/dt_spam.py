import numpy as np

from src.DecisionTree import DecisionTree


def accuracy(Y, pred):
    n = Y.shape[0]
    return np.sum(Y == pred) / n


k = 5
data = np.genfromtxt('../data/spambase/spambase.data', delimiter=',')
l = data.shape[0]
indexes = np.arange(l)
np.random.shuffle(indexes)

tp = 0
tn = 0
fp = 0
fn = 0

cnt_cutoff = 3
gain_cutoff = 0.1
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

    tree = DecisionTree()
    tree.train(X_train, Y_train, cnt_cutoff, gain_cutoff)

    pred_train = tree.run(X_train)
    train_ac.append(accuracy(Y_train.T, pred_train.flatten()))

    pred_test = tree.run(X_test)
    test_ac.append(accuracy(Y_test.T, pred_test.flatten()))

    tp += np.logical_and(Y_test.flatten(), pred_test.flatten()).sum()
    tn += np.logical_and(np.logical_not(Y_test.flatten()), np.logical_not(pred_test.flatten())).sum()
    fp += np.logical_and(Y_test.flatten(), np.logical_not(pred_test.flatten())).sum()
    fn += np.logical_and(np.logical_not(Y_test.flatten()), pred_test.flatten()).sum()

print('min leaf count:', cnt_cutoff, 'min info gain:', gain_cutoff)
print(' train accuracy: ', np.mean(train_ac))
print(' test accuracy: ', np.mean(test_ac))
print(tp, fp, fn, tn)

