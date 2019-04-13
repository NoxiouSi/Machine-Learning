import numpy as np
from NeuralNet import NeuralNet, sigmoid

x = np.eye(8)
y = np.array(x)
nn = NeuralNet([8,3,8], 1)

nn.train(x,y)
print(nn.predict(x))
# print(np.argmax(nn.predict(x), axis=1))
# print(sigmoid(x.dot(nn.w[0])))
