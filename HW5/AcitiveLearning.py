from boosting import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    T = 100

    data = np.genfromtxt('../HW1/data/spambase/spambase.data', delimiter=',')
    l = data.shape[0]
    indexes = np.arange(l)

    label_perc = 5

    training_idx = np.random.choice(indexes, int(l*0.05))

    test_error = []
    labeled_percent = []

    print('Active Learning:')
    while label_perc < 50:
        testing_idx = indexes[np.array([i not in training_idx for i in indexes])]
        train_data = data[training_idx, :]
        test_data = data[testing_idx, :]
        X_train = train_data[:, :-1]
        Y_train = train_data[:, -1] * 2 - 1
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1] * 2 - 1

        bst = Booster(T, OptimalDecisionStump)
        bst.fit(X_train, Y_train)
        pred = bst.predict(X_test)
        err = np.not_equal(pred, Y_test).sum() / Y_test.shape[0]
        print('labeled', label_perc, '% data, error:', err)
        test_error.append(err)
        labeled_percent.append(label_perc)

        pred_prob = bst.predict_prob(X_test)
        conf = np.abs(pred_prob)

        k = int(l*0.02)
        data_to_label = testing_idx[np.argpartition(conf, k)[:k]]
        training_idx = np.append(training_idx, data_to_label)
        label_perc += 2

    plt.figure()
    line1, = plt.plot(labeled_percent, test_error, 'r')

    print('Random Labeling data:')

    random_percent = [5, 10, 15, 20, 30, 50]
    random_error = []
    for c in random_percent:
        training_idx = np.random.choice(indexes, int(l * c / 100))
        testing_idx = indexes[np.array([i not in training_idx for i in indexes])]
        train_data = data[training_idx, :]
        test_data = data[testing_idx, :]
        X_train = train_data[:, :-1]
        Y_train = train_data[:, -1] * 2 - 1
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1] * 2 - 1


        bst = Booster(T, OptimalDecisionStump)
        bst.fit(X_train, Y_train)
        pred = bst.predict(X_test)
        err = np.not_equal(pred, Y_test).sum() / Y_test.shape[0]
        print('labeled', c, '% data, error:', err)
        random_error.append(err)

    line2, = plt.plot(random_percent, random_error, 'b')
    plt.legend([line1, line2], ['Active Learning', 'Random'])
    # plt.show()
    plt.savefig('q3result/active vs random.png')

