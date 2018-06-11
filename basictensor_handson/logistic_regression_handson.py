#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)
# plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label="Positive")
# plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label="Negative")
# plt.legend()
# plt.show()
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
y_moons_column_vector = y_moons.reshape(-1, 1)

test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]

#下面这个生成random_batch的方法有待改进,这样并不会所有的数据都被训练
def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

# X_batch, y_batch = random_batch(X_train, y_train, 5)

n_inputs = 2

X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed=42), name="theta")
logits = tf.matmul(X, theta, name="logits")
# y_proba = 1 / (1 + tf.exp(-logits))
# epsilon = 1e-7  # to avoid an overflow when computing the log
# loss = -tf.reduce_mean(y * tf.log(y_proba + epsilon) + (1 - y) * tf.log(1 - y_proba + epsilon))
y_proba = tf.sigmoid(logits)
loss = tf.losses.log_loss(y, y_proba)  # uses epsilon = 1e-7 by default

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 500
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val = loss.eval({X: X_test, y: y_test})
        if epoch % 100 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)
        if epoch % 200 == 0:
            y_proba_val_dev = y_proba.eval(feed_dict={X: X_test, y: y_test})
            y_pred_dev = (y_proba_val_dev >= 0.5)
            print("precision:" + str(precision_score(y_test, y_pred_dev)))
            print("recall:" + str(recall_score(y_test, y_pred_dev)))
    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
    y_pred = (y_proba_val >= 0.5)
#To_do,使用tensorflow的metrics修改这边的metrics,tensorflow的metrics有坑，在于他要建立一个自己的graph
print("all performance")
print("precision:" + str(precision_score(y_test, y_pred)))
print("recall:" + str(recall_score(y_test, y_pred)))