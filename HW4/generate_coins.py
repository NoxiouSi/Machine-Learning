import numpy as np

M = 1000
K = 10

n = 3
q = [0.2, 0.5, 0.9]
pi = [0.1, 0.3, 0.6]

with open('data/2coin.txt', 'w+') as f:
    for i in range(M):
        seq = []
        coin = np.random.rand()
        for j in range(n):
            if coin < pi[j]:
                break
            else:
                coin -= pi[j]
        for k in range(K):
            flip = np.random.rand()
            if flip < q[j]:
                seq.append('1')
            else:
                seq.append('0')

        f.write(','.join(seq)+'\n')
