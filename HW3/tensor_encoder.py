import tensorflow as tf
import numpy as np

input_unit_num = 8
hidden_unit_num = 3
output_unit_num = 8

train_x = np.eye(8)
train_y = np.eye(8)

x = tf.placeholder(tf.float32, [None, input_unit_num])
y = tf.placeholder(tf.float32, [None, output_unit_num])

epochs = 100000
batch_size = 8
learning_rate = 1

weights = {
    'hidden': tf.Variable(tf.random_normal([input_unit_num, hidden_unit_num])),
    'output': tf.Variable(tf.random_normal([hidden_unit_num, output_unit_num]))
}

hidden_layer = tf.nn.sigmoid(tf.matmul(x, weights['hidden']))
output_layer = tf.nn.sigmoid(tf.matmul(hidden_layer, weights['output']))

cost = tf.losses.mean_squared_error(labels=y, predictions=output_layer)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)

    epoch = 1
    while True:
        _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})

        if epoch % 100 is 0:
            print("Epoch:", epoch , "cost =", "{:.5f}".format(c))
        if c < 0.02:
            break
        epoch += 1
    print(sess.run(output_layer, feed_dict={x: train_x}))