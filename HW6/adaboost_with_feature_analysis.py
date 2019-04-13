import numpy as np
from sklearn.metrics import roc_auc_score

from boosting import AdaBoost, OptimalDecisionStump

if __name__ == '__main__':

    k = 10
    T = 300

    print('Original Dataset')
    print('Reading data')
    data = np.genfromtxt('../HW1/data/spambase/spambase.data', delimiter=',')
    x = data[:, :-1]
    y = data[:, -1]

    print('Training model')
    # bst = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, splitter='best', criterion='entropy'), n_estimators=T, algorithm='SAMME.R')
    bst = AdaBoost(T, OptimalDecisionStump)
    bst.fit(x, y, test_data=(x,y))
    print('Analyzing')
    score = np.zeros(x.shape[1])
    # for i in range(T):
    # sub_tree = bst.estimators_[i]
    # score[sub_tree.tree_.feature] += (y * bst.estimator_weights_[i] * sub_tree.predict(x)).sum()
    for i in range(T):
        sub_tree = bst.classifiers[i]
        score[sub_tree.feature] += (y * bst.alpha[i] * sub_tree.predict(x)).sum()
    score /= score.sum()
    print('Top 15 features', np.argsort(score)[::-1][:15])
    pred = bst.predict(x)
    print("Accuracy:", np.mean(np.equal(pred, y)))

    print()
    print('Polluted Dataset')
    print('Reading data')
    train_x = np.genfromtxt('spam_polluted/train_feature.txt', delimiter=' ')
    train_y = np.genfromtxt('spam_polluted/train_label.txt', delimiter=' ')
    test_x = np.genfromtxt('spam_polluted/test_feature.txt', delimiter=' ')
    test_y = np.genfromtxt('spam_polluted/test_label.txt', delimiter=' ')

    print('Training model')
    # bst = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, splitter='best', criterion='entropy'), n_estimators=T, algorithm='SAMME.R')
    bst = AdaBoost(T, OptimalDecisionStump)
    bst.fit(train_x, train_y, test_data=(test_x, test_y))
    print('Analyzing')
    score = np.zeros(train_x.shape[1])
    # for i in range(T):
    #     sub_tree = bst.estimators_[i]
    #     score[sub_tree.tree_.feature] += (train_y * bst.estimator_weights_[i] * sub_tree.predict(train_x)).sum()
    for i in range(T):
        sub_tree = bst.classifiers[i]
        score[sub_tree.feature] += (train_y * bst.alpha[i] * sub_tree.predict(train_x)).sum()
    score /= score.sum()
    print('Top 15 features', np.argsort(score)[::-1][:15])

    pred = bst.predict(test_x)
    print("Accuracy:", np.mean(np.equal(pred, test_y)))
