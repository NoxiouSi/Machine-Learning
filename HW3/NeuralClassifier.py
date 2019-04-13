import numpy as np


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    s = np.exp(x)
    return s / np.sum(s, axis=1, keepdims=True)


class NeuralClassifier:

    def __init__(self, n_layers, LAMBDA=0.1):
        self.n_layers = n_layers
        self.w = None
        self.mean = None
        self.var = None
        self.LAMBDA = LAMBDA

    def normalize(self, X):
        _X = np.array(X)
        self.mean = _X.mean(axis=0)
        _X = _X - self.mean
        self.var = _X.var(axis=0)
        _X = _X / self.var
        return _X

    def train(self, X, Y):

        m, n = X.shape
        _, o = Y.shape

        if n is not self.n_layers[0] or o is not self.n_layers[-1]:
            print(m, n, o)
            raise ValueError

        self.w = []
        for i in range(0, len(self.n_layers) - 1):
            self.w.append(np.random.rand(self.n_layers[i], self.n_layers[i + 1]))

        epoch = 1
        last_j = np.inf
        while True:
            # forward
            a = []
            z = []
            a.append(X)
            for t in self.w:
                z.append(a[-1].dot(t))
                a.append(relu(z[-1]))
            p = softmax(z[-1])
            J = -np.log(np.sum(Y * p, axis=1)).sum()
            if epoch % 100 is 0:
                print('epoch', epoch, ':', J)
            # end condition
            if last_j - J < 0.000001:
            # if J < 2.3:
                # if last_j - J < 0:
                #     print('boom')
                break
            last_j = J

            delta = []
            dw = []
            delta.insert(0, (Y - softmax(a[-1])))
            for i in range(len(self.n_layers) - 2, -1, -1):
                dw.insert(0, a[i].T.dot(delta[0]))
                if i > 0:
                    delta.insert(0, delta[0].dot(self.w[i].T) * (a[i - 1].dot(self.w[i - 1]) > 0))
            for i in range(len(self.w)):
                self.w[i] += self.LAMBDA * dw[i]

            epoch += 1

    def predict(self, X):
        a = []
        z = []
        a.append(X)
        for t in self.w:
            z.append(a[-1].dot(t))
            a.append(relu(z[-1]))
        p = softmax(z[-1])
        return p
