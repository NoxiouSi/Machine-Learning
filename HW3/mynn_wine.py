import numpy as np
from NeuralNet import NeuralNet
from sklearn.preprocessing import LabelBinarizer

train = np.loadtxt('./wine/train_wine.csv', delimiter=',')
lb = LabelBinarizer()
train_y = lb.fit_transform(train[:,0])
train_x = train[:,1:]
mean = np.mean(train_x, axis=0)
train_x -= mean
var = np.var(train_x, axis=0)
train_x /= var

nn = NeuralNet([13,13,3], 0.01)

nn.train(train_x,train_y)

test = np.loadtxt('./wine/test_wine.csv', delimiter=',')
test_y = test[:,0]
test_x = test[:,1:]
test_x -= mean
test_x /= var

y_pred = np.argmax(nn.predict(test_x), axis=1)
accu = np.sum(test_y == y_pred+1) / len(test_y)
print(accu)
