from nb import NB
from nb_gaussian import NBGau
from nb_four_bin import NB4Bin
from nb_nine_bin import NBNBin

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('../HW1/data/spambase/spambase.data', delimiter=',')

k = 10
l = data.shape[0]
indexes = np.arange(l)
np.random.shuffle(indexes)

pred_nb = []
pred_nb_gau = []
pred_nb_four_bin = []
pred_nb_nine_bin = []

for i in range(k):
    idx = np.arange(l)
    cv_set = indexes[int(i * l / k):int((i + 1) * l / k)]
    train_data = data[np.logical_not(np.isin(idx, cv_set)), :]
    test_data = data[cv_set, :]
    X_train = train_data[:, :-1]
    Y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1]

    m = len(Y_test)

    nb = NB()
    nb.train(X_train, Y_train)
    diff = nb.predict(X_test, raw_diff=True)
    for i in range(m):
        pred_nb.append((diff[i], Y_test[i]))

    nb = NBGau()
    nb.train(X_train, Y_train)
    diff = nb.predict(X_test, raw_diff=True)
    for i in range(m):
        pred_nb_gau.append((diff[i], Y_test[i]))

    nb = NB4Bin()
    nb.train(X_train, Y_train)
    diff = nb.predict(X_test, raw_diff=True)
    for i in range(m):
        pred_nb_four_bin.append((diff[i], Y_test[i]))

    nb = NBNBin(9)
    nb.train(X_train, Y_train)
    diff = nb.predict(X_test, raw_diff=True)
    for i in range(m):
        pred_nb_nine_bin.append((diff[i], Y_test[i]))

pred_nb.sort(key=lambda x: x[0])  # sort by diff value
pred_nb_gau.sort(key=lambda x: x[0])  # sort by diff value
pred_nb_four_bin.sort(key=lambda x: x[0])  # sort by diff value
pred_nb_nine_bin.sort(key=lambda x: x[0])  # sort by diff value

pos = sum(1 for p, lab in pred_nb if lab == 1)
neg = sum(1 for p, lab in pred_nb if lab == 0)


def create_curve(pred):
    tpr = [0]
    fpr = [0]
    tp = 0
    fp = 0
    for p, lab in pred:
        if lab == 1:
            fp += 1
        else:
            tp += 1
        tpr.append(tp / neg)
        fpr.append(fp / pos)
    return tpr, fpr


def auc(x, y):
    area = 0
    for i in range(len(x)-1):
        area += (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2
    return area


fig = plt.figure()

tpr, fpr = create_curve(pred_nb)
print('auc(Naive Bayes):', auc(fpr, tpr))
l1, = plt.plot(fpr, tpr, color='r', label='Naive Bayes')
tpr, fpr = create_curve(pred_nb_gau)
print('auc(Naive Bayes - Gaussian):', auc(fpr, tpr))
l2, = plt.plot(fpr, tpr, color='b', label='Naive Bayes - Gaussian')
tpr, fpr = create_curve(pred_nb_four_bin)
print('auc(Naive Bayes - 4 bins):', auc(fpr, tpr))
l3, = plt.plot(fpr, tpr, color='g', label='Naive Bayes - 4 bins')
tpr, fpr = create_curve(pred_nb_nine_bin)
print('auc(Naive Bayes - 9 bins):', auc(fpr, tpr))
l4, = plt.plot(fpr, tpr, color='y', label='Naive Bayes - 9 bins')

plt.legend(handles=[l1, l2, l3, l4])
fig.show()
fig.savefig('ROC curve')
