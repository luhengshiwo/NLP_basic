#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
from sklearn.metrics import precision_score, recall_score
import os

mnist = input_data.read_data_sets("../../../data/")
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

X_train2_full = X_train[y_train >= 5]
X_test2 = X_test[y_test >= 5]
y_train2_full = y_train[y_train >= 5] - 5
y_test2 = y_test[y_test >= 5] - 5
X_valid2_full = mnist.validation.images[mnist.validation.labels >= 5]
y_valid2_full = mnist.validation.labels[mnist.validation.labels >= 5] - 5

n_inputs = 28 * 28  # MNIST
n_hidden = 100
n_outputs = 5
n_epochs = 20
batch_size = 50
learning_rate = 0.01
best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50


checkpoint_path = "./tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_logreg_model_0_to_4"


# restore_saver = tf.train.import_meta_graph("./my_logreg_model_0_to_4.meta")

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
he_init = tf.contrib.layers.variance_scaling_initializer()

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden, name="hidden1", kernel_initializer=he_init,
                              activation=tf.nn.elu)
    hidden1_stop = tf.stop_gradient(hidden1)
    hidden2 = tf.layers.dense(hidden1_stop, n_hidden, name="hidden2", kernel_initializer=he_init,
                              activation=tf.nn.elu)
    hidden3 = tf.layers.dense(hidden2, n_hidden, name="hidden3", kernel_initializer=he_init,
                              activation=tf.nn.elu)
    hidden4 = tf.layers.dense(hidden3, n_hidden, name="hidden4", kernel_initializer=he_init,
                              activation=tf.nn.elu)
    
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="hidden[1234]")  # regular expression
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
restore_saver = tf.train.Saver(reuse_vars_dict)  # to restore layers 1-3


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('log_accuracy', accuracy)


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


logdir = log_dir("logreg")

init = tf.global_variables_initializer()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
five_frozen_saver = tf.train.Saver()


def sample_n_instances_per_class(X, y, n=100):
    Xs, ys = [], []
    for label in np.unique(y):
        idx = (y == label)
        Xc = X[idx][:n]
        yc = y[idx][:n]
        Xs.append(Xc)
        ys.append(yc)
    return np.concatenate(Xs), np.concatenate(ys)


X_train2, y_train2 = sample_n_instances_per_class(
    X_train2_full, y_train2_full, n=100)
X_valid2, y_valid2 = sample_n_instances_per_class(
    X_valid2_full, y_valid2_full, n=30)

import time

n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

# with tf.Session() as sess:
#     init.run()
#     restore_saver.restore(sess, "./my_logreg_model_0_to_4")
#     h1_cache = sess.run(hidden1, feed_dict={X: mnist.train.images})
#     h1_cache_test = sess.run(hidden1, feed_dict={X: mnist.test.images}) 
#     t0 = time.time()
        
#     for epoch in range(n_epochs):
#         rnd_idx = np.random.permutation(len(X_train2))
#         for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
#             X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid2, y: y_valid2})
#         if loss_val < best_loss:
#             save_path = five_frozen_saver.save(sess, "./my_mnist_model_5_to_9_five_frozen")
#             best_loss = loss_val
#             checks_without_progress = 0
#         else:
#             checks_without_progress += 1
#             if checks_without_progress > max_checks_without_progress:
#                 print("Early stopping!")
#                 break
#         print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
#             epoch, loss_val, best_loss, acc_val * 100))

#     t1 = time.time()
#     print("Total training time: {:.1f}s".format(t1 - t0))

with tf.Session() as sess:
    five_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_five_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test2, y: y_test2})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
