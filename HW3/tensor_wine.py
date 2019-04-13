import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer

input_unit_num = 13
hidden_unit_num = 13
output_unit_num = 3

train = np.loadtxt('./wine/train_wine.csv', delimiter=',')
lb = LabelBinarizer()
train_y = lb.fit_transform(train[:,0])
train_x = train[:,1:]
mean = np.mean(train_x, axis=0)
train_x -= mean
var = np.var(train_x, axis=0)
train_x /= var


x = tf.placeholder(tf.float32, [None, input_unit_num])
y = tf.placeholder(tf.float32, [None, output_unit_num])

learning_rate = 1

weights = {
    'hidden': tf.Variable(tf.random_normal([input_unit_num, hidden_unit_num])),
    'output': tf.Variable(tf.random_normal([hidden_unit_num, output_unit_num]))
}

hidden_layer = tf.nn.sigmoid(tf.matmul(x, weights['hidden']))
output_layer = tf.nn.sigmoid(tf.matmul(hidden_layer, weights['output']))

cost = tf.losses.mean_squared_error(labels=train_y, predictions=output_layer)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)

    epoch = 1
    while True:
        _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})

        if epoch % 100 is 0:
            print("Epoch:", epoch , "cost =", "{:.5f}".format(c))
        if c < 0.01:
            break
        epoch += 1

    test = np.loadtxt('./wine/test_wine.csv', delimiter=',')
    test_y = test[:, 0]
    test_x = test[:, 1:]
    test_x -= mean
    test_x /= var

    y_pred = np.argmax(sess.run(output_layer, feed_dict={x: test_x}), axis=1)
    accu = np.sum(y_pred+1 == test_y) / len(test_y)
    print(accu)