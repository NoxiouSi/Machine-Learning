import numpy as np
from scipy.stats import multivariate_normal

epoches = 30

x = np.genfromtxt('data/2gaussian.txt')

m, d = x.shape

part0 = range(0, int(m / 2))
part1 = range(int(m / 2), m)
mu0 = x[part0].mean(axis=0)
cov0 = np.cov(x[part0].T)
mu1 = x[part1].mean(axis=0)
cov1 = np.cov(x[part1].T)
z = np.zeros((2, m))
z[0][part0] = 1
z[1][part1] = 1
pi0 = 0.5
pi1 = 0.5

for i in range(epoches):
    # E step
    z[0] = pi0 * multivariate_normal.pdf(x, mean=mu0, cov=cov0)
    z[1] = pi1 * multivariate_normal.pdf(x, mean=mu1, cov=cov1)
    z /= z.sum(axis=0)

    # M step
    mu0 = (z[0].dot(x)) / z[0].sum()
    mu1 = (z[1].dot(x)) / z[1].sum()

    cov0 = ((x - mu0).T * z[0, None]).dot(x - mu0) / z[0].sum()
    cov1 = ((x - mu1).T * z[1, None]).dot(x - mu1) / z[1].sum()

    pi0 = z[0].sum() / m
    pi1 = z[1].sum() / m

    print('epoch', i)
    print(mu0)
    print(cov0)
    print(mu1)
    print(cov1)
