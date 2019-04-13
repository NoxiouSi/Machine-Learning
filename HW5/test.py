import numpy as np

d = 2
X = np.zeros((50,2))
X[:10,0] = 1
X[:,1] = np.arange(50, 0, step=-1)
weight = np.ones(50) / 50
Y = np.zeros(50)
Y[25:] = 1
Y[:25] = -1
Y[10] = 1
Y[20] = 1
Y[30] = -1

from boosting import OptimalDecisionStump

clf = OptimalDecisionStump()
clf.fit(X, weight, Y)
print((clf.predict(X)==Y).sum() / 50)