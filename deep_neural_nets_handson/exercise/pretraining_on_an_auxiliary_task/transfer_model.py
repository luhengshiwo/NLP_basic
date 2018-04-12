#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
from sklearn.metrics import precision_score, recall_score
import os
from split import generate_batch

mnist = input_data.read_data_sets("../../../data/")
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

X_train1 = X_train
X_test = X_test
y_train1 = y_train
y_test = y_test
X_valid1 = mnist.validation.images
y_valid1 = mnist.validation.labels

n_inputs = 28 * 28  # MNIST
n_hidden = 100
n_outputs = 10
n_epochs = 100
batch_size = 500
learning_rate = 0.01
best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50
checkpoint_path = "./tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_logreg_model_0_to_4"


X1 = tf.placeholder(tf.float32, shape=(None,n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
he_init = tf.contrib.layers.variance_scaling_initializer()

with tf.name_scope("dnn1"):
    hidden1_1 = tf.layers.dense(X1, n_hidden, name="hidden1_1", kernel_initializer=he_init,
                              activation=tf.nn.elu)
    hidden2_1 = tf.layers.dense(hidden1_1, n_hidden, name="hidden2_1", kernel_initializer=he_init,
                              activation=tf.nn.elu)
    hidden3_1 = tf.layers.dense(hidden2_1, n_hidden, name="hidden3_1", kernel_initializer=he_init,
                              activation=tf.nn.elu)
    hidden4_1 = tf.layers.dense(hidden3_1, n_hidden, name="hidden4_1", kernel_initializer=he_init,
                              activation=tf.nn.elu)
    hidden_stop = tf.stop_gradient(hidden4_1)
# with tf.name_scope("dnn2"):
#     hidden1_2 = tf.layers.dense(X2, n_hidden, name="hidden1_2", kernel_initializer=he_init,
#                               activation=tf.nn.elu)
#     hidden2_2 = tf.layers.dense(hidden1_2, n_hidden, name="hidden2_2", kernel_initializer=he_init,
#                               activation=tf.nn.elu)
#     hidden3_2 = tf.layers.dense(hidden2_2, n_hidden, name="hidden3_2", kernel_initializer=he_init,
#                               activation=tf.nn.elu)
#     hidden4_2 = tf.layers.dense(hidden3_2, n_hidden, name="hidden4_2", kernel_initializer=he_init,
#                               activation=tf.nn.elu)

with tf.name_scope("logits"):
    # hidden4 = tf.concat([hidden4_1,hidden4_2],axis=1)
    # hidden = tf.layers.dense(hidden_stop, units=10, activation=tf.nn.elu, kernel_initializer=he_init)
    logits = tf.layers.dense(hidden_stop, n_outputs, name="outputs",kernel_initializer=he_init)
    y_proba = tf.nn.sigmoid(logits)

with tf.name_scope("loss"):
    # y_as_float = tf.cast(y, tf.float32)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    y_pred = tf.cast(tf.greater_equal(logits, 0), tf.int32)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('log_accuracy', accuracy)


reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="hidden[1_12_13_14_1]")  # regular expression
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
restore_saver = tf.train.Saver(reuse_vars_dict)  # to restore layers 1-3

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


logdir = log_dir("logreg")

init = tf.global_variables_initializer()
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
five_frozen_saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_logreg_model_0_to_4")
    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_valid1))
        for rnd_indices in np.array_split(rnd_idx, len(X_valid1) // batch_size):
            X_batch, y_batch = X_valid1[rnd_indices], y_valid1[rnd_indices]
            sess.run(training_op, feed_dict={X1: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X1: X_valid1, y: y_valid1})
        if loss_val < best_loss:
            save_path = five_frozen_saver.save(sess, "./my_mnist_model_final_frozen")
            best_loss = loss_val
            epochs_without_progress = 0
        else:
            epochs_without_progress += 1
            if epochs_without_progress > max_epochs_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))




# with tf.Session() as sess:
#     saver.restore(sess, final_model_path)
#     acc_test = accuracy.eval(feed_dict={X: X_test1, y: y_test1})
#     print("Final test accuracy: {:.2f}%".format(acc_test * 100))


# tensorboard --logdir=tf_logs


#注意最后损失函数的使用