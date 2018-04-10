#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/")
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300  
n_hidden2 = 50  
n_hidden3 = 50  
n_hidden4 = 20  
n_outputs = 10  

n_epochs = 4
batch_size = 50

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):       
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               decay_steps, decay_rate)
    #  decayed_learning_rate = learning_rate *
                        #   decay_rate ^ (global_step / decay_steps)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                    #    momentum=0.9, use_nesterov=True)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
    #                                   momentum=0.9, decay=0.9, epsilon=1e-10)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # 注意，AdaGrad RMSProp Adam 不需要learning_rate_decay
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")


#learning_rate_decay 一般只在momentum上使用