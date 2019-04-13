import numpy as np

epoches = 200
K = 3

x = np.genfromtxt('data/2coin.txt', delimiter=',')

m, d = x.shape

q = np.arange(1, K + 1) / (K + 1)
z = np.zeros((K, m))
for k in range(K):
    z[k][int(m / K * k):int(m / K * (k + 1))] = 1
pi = np.ones(K) / K

for i in range(epoches):
    # E step
    for k in range(K):
        z[k] = pi[k] * np.exp(np.sum(x.dot(np.log(q[k])) + (1 - x).dot(np.log(1 - q[k])), axis=1))
    z /= z.sum(axis=0)

    # M step
    for k in range(K):
        q[k] = np.average(z[k].dot(x) / z[k].sum())

    for k in range(K):
        pi[k] = z[k].sum() / m

    print('epoch', i)
    for k in range(K):
        print('π', k, '{0:.2f}'.format(pi[k]))
        print('q0', k, '{0:.2f}'.format(q[k]))
