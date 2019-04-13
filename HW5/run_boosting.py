from boosting import *
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


def read_data(dataset, encoder=None):
    cat = []
    config_file = os.path.join('data', dataset, dataset+'.config')
    data_file =  os.path.join('data', dataset, dataset+'.data')
    with open(config_file) as f:
        head = f.readline()
        m, conti, dis = (int(w) for w in head.split())
        for i in range(conti + dis + 1):
            line = f.readline()
            cat.append(int(line.split()[0]) > 0)
    data = pd.read_csv(data_file, delimiter='\\s+', header=None, na_values='?')
    data = data.dropna()
    for col in range(data.shape[1]):
        if cat[col]:
            data[col].fillna(value='?', inplace=True)
        else:
            data[col].fillna(value=data[col].mean())
    if encoder is None:
        encoder = OneHotEncoder()
    res = data.iloc[:, np.logical_not(cat)].values
    res = np.append(res, encoder.fit_transform(data.iloc[:, cat].values).toarray(), axis=1)
    return res


if __name__ == '__main__':

    T = 100
    data = read_data('vote')

    kf = KFold(n_splits=5, shuffle=True)

    for c in np.array([5, 10, 15, 20, 30, 50, 80]):
        print('using ' + str(c) + '% as training:')
        train_error = 0
        test_error = 0
        i = 1
        pred_prob = np.array([])
        true_label = np.array([])
        for training_set, test_set in kf.split(data):
            training_data = data[np.random.choice(training_set, int(data.shape[0] * c / 100))]
            test_data = data[test_set, :]

            X_train = training_data[:, :-1]
            Y_train = training_data[:, -1] * 2 - 1
            X_test = test_data[:, :-1]
            Y_test = test_data[:, -1] * 2 - 1

            # print(X_train)
            # print(Y_train)

            bst = Booster(T, OptimalDecisionStump)
            bst.fit(X_train, Y_train)

            Y_pred = bst.predict(X_train)
            train_e = np.not_equal(Y_pred, Y_train).sum()

            Y_pred = bst.predict(X_test)
            test_e = np.not_equal(Y_pred, Y_test).sum()
            print('epoch', i, 'training error', train_e, 'test error', test_e)
            train_error += train_e
            test_error += test_e

            pred_prob = np.append(pred_prob, bst.predict_prob(X_test))
            true_label = np.append(true_label, Y_test)

            i += 1
        m = data.shape[0]
        print('training error:', train_error / m, 'test error:', test_error / m,
              'auc:', roc_auc_score(true_label, pred_prob))
        print()
