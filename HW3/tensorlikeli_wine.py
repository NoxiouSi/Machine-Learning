import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer

input_unit_num = 13
hidden_unit1_num = 3
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

learning_rate = 0.0001

weights = {
    'hidden1': tf.Variable(tf.random_normal([input_unit_num, hidden_unit1_num])),
    'output': tf.Variable(tf.random_normal([hidden_unit1_num, output_unit_num]))
}

hidden_layer1 = tf.nn.relu(tf.matmul(x, weights['hidden1']))
output_layer = tf.matmul(hidden_layer1, weights['output'])
softmax_output = tf.nn.softmax(output_layer, axis=1)

cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(train_y), logits=output_layer)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

test = np.loadtxt('./wine/test_wine.csv', delimiter=',')
test_y = test[:, 0]
test_x = test[:, 1:]
test_x -= mean
test_x /= var

with tf.Session() as sess:
    sess.run(init)

    epoch = 1
    last_c = np.inf
    while True:
        _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})

        c = np.sum(c)
        if epoch % 100 is 0:
            print("Epoch:", epoch , "cost =", c)

        if last_c - c < 0.000001:
        # if c < 1:
        #     if last_c - c < 0:
        #         print('boom')
            break
        last_c = c
        epoch += 1

    y_pred = np.argmax(sess.run(softmax_output, feed_dict={x: test_x}), axis=1)
    accu = np.sum(y_pred+1 == test_y) / len(test_y)
    print(accu)