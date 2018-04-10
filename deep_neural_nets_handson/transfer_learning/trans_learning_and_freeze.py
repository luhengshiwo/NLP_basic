#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../data/")
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300  # reused
n_hidden2 = 50  # reused
n_hidden3 = 50  # reused
n_hidden4 = 20  # new!
n_outputs = 10  # new!
learning_rate = 0.01
n_epochs = 4
batch_size = 50

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(
        X, n_hidden1, activation=tf.nn.relu, name="hidden1")       # reused
    hidden2 = tf.layers.dense(
        hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")  # reused
    hidden3 = tf.layers.dense(
        hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")  # reused
    hidden4 = tf.layers.dense(
        hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")  # new!
    logits = tf.layers.dense(
        hidden4, n_outputs, name="outputs")                         # new!

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")


#是否freeze前面几层训练出来的参数：
# with tf.name_scope("train"):
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     training_op = optimizer.minimize(loss)

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope="hidden[34]|outputs")
    training_op = optimizer.minimize(loss, var_list=train_vars)

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="hidden[123]")  # regular expression
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
restore_saver = tf.train.Saver(reuse_vars_dict)  # to restore layers 1-3

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")

    # not shown in the book
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):  # not shown
            X_batch, y_batch = mnist.train.next_batch(
                batch_size)      # not shown
            sess.run(training_op, feed_dict={
                     X: X_batch, y: y_batch})  # not shown
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,  # not shown
                                                y: mnist.test.labels})  # not shown
        # not shown
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_new_model_final.ckpt")

# 使用之前训练好的layers，加上新定义的层，去做transfer_learning,
# 和transfer_learning_handson.py的不同之处在于我知道源代码
# 本文件还包含一个freeze前几层的一个方法，重用hiddenlayer123,freeze掉hidden12(可训练的是3，4，output)
#另外一种方法是：
# with tf.name_scope("dnn"):
#     hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
#                               name="hidden1") # reused frozen
#     hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
#                               name="hidden2") # reused frozen
#     hidden2_stop = tf.stop_gradient(hidden2)
#     hidden3 = tf.layers.dense(hidden2_stop, n_hidden3, activation=tf.nn.relu,
#                               name="hidden3") # reused, not frozen
#     hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu,
#                               name="hidden4") # new!
#     logits = tf.layers.dense(hidden4, n_outputs, name="outputs") # new!

# with tf.name_scope("train"):
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     training_op = optimizer.minimize(loss)

# 注意到 hidden2_stop的建立和 训练loss的方法
