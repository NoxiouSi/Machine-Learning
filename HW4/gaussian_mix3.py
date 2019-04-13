import numpy as np
from scipy.stats import multivariate_normal

epoches = 80

x = np.genfromtxt('data/3gaussian.txt')

m, d = x.shape

part0 = range(0, int(m / 3))
part1 = range(int(m / 3), int(m * 2 / 3))
part2 = range(int(m * 2 / 3), m)
mu0 = x[part0].mean(axis=0)
cov0 = np.cov(x[part0].T)
mu1 = x[part1].mean(axis=0)
cov1 = np.cov(x[part1].T)
mu2 = x[part2].mean(axis=0)
cov2 = np.cov(x[part2].T)
z = np.zeros((3, m))
z[0][part0] = 1
z[1][part1] = 1
z[2][part2] = 1
pi0 = 1 / 3
pi1 = 1 / 3
pi2 = 1 / 3

for i in range(epoches):
    # E step
    z[0] = pi0 * multivariate_normal.pdf(x, mean=mu0, cov=cov0)
    z[1] = pi1 * multivariate_normal.pdf(x, mean=mu1, cov=cov1)
    z[2] = pi2 * multivariate_normal.pdf(x, mean=mu2, cov=cov2)
    z /= z.sum(axis=0)

    # M step
    mu0 = (z[0].dot(x)) / z[0].sum()
    mu1 = (z[1].dot(x)) / z[1].sum()
    mu2 = (z[2].dot(x)) / z[2].sum()

    cov0 = ((x - mu0).T * z[0, None]).dot(x - mu0) / z[0].sum()
    cov1 = ((x - mu1).T * z[1, None]).dot(x - mu1) / z[1].sum()
    cov2 = ((x - mu2).T * z[2, None]).dot(x - mu2) / z[2].sum()

    pi0 = z[0].sum() / m
    pi1 = z[1].sum() / m
    pi2 = z[2].sum() / m

    print('epoch', i)
    print(mu0)
    print(cov0)
    print(mu1)
    print(cov1)
    print(mu2)
    print(cov2)
