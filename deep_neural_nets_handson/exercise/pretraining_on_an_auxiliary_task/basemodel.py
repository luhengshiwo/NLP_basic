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
n_outputs = 1
n_epochs = 100
batch_size = 500
learning_rate = 0.01
best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50
checkpoint_path = "./tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_logreg_model_compare"


X = tf.placeholder(tf.float32, shape=(None, 2, n_inputs), name="X")
X1, X2 = tf.unstack(X, axis=1)
y = tf.placeholder(tf.int32, shape=(None, 1), name="y")
he_init = tf.contrib.layers.variance_scaling_initializer()


# def dnn(inputs, n_hidden_layers=5, n_neurons=100, name=None,
#         activation=tf.nn.elu, initializer=he_init):
#     with tf.variable_scope(name, "dnn"):
#         for layer in range(1,n_hidden_layers):
#             inputs = tf.layers.dense(inputs, n_neurons, activation=activation,
#                                      kernel_initializer=initializer,
#                                      name="hidden%d" % (layer + 1))
#         return inputs


# dnn1 = dnn(X1, name='DNN_A')
# dnn2 = dnn(X2, name="DNN_B")

with tf.name_scope("dnn1"):
    hidden1_1 = tf.layers.dense(X1, n_hidden, name="hidden1_1", kernel_initializer=he_init,
                                activation=tf.nn.elu)
    hidden2_1 = tf.layers.dense(hidden1_1, n_hidden, name="hidden2_1", kernel_initializer=he_init,
                                activation=tf.nn.elu)
    hidden3_1 = tf.layers.dense(hidden2_1, n_hidden, name="hidden3_1", kernel_initializer=he_init,
                                activation=tf.nn.elu)
    hidden4_1 = tf.layers.dense(hidden3_1, n_hidden, name="hidden4_1", kernel_initializer=he_init,
                                activation=tf.nn.elu)

with tf.name_scope("dnn2"):
    hidden1_2 = tf.layers.dense(X2, n_hidden, name="hidden1_2", kernel_initializer=he_init,
                                activation=tf.nn.elu)
    hidden2_2 = tf.layers.dense(hidden1_2, n_hidden, name="hidden2_2", kernel_initializer=he_init,
                                activation=tf.nn.elu)
    hidden3_2 = tf.layers.dense(hidden2_2, n_hidden, name="hidden3_2", kernel_initializer=he_init,
                                activation=tf.nn.elu)
    hidden4_2 = tf.layers.dense(hidden3_2, n_hidden, name="hidden4_2", kernel_initializer=he_init,
                                activation=tf.nn.elu)

with tf.name_scope("logits"):
    hidden4 = tf.concat([hidden4_1,hidden4_2],axis=1)
    # hidden4 = tf.concat([dnn1, dnn2], axis=1)
    hidden = tf.layers.dense(
        hidden4, units=10, activation=tf.nn.elu, kernel_initializer=he_init)
    logits = tf.layers.dense(
        hidden, n_outputs, name="outputs", kernel_initializer=he_init)
    y_proba = tf.nn.sigmoid(logits)

with tf.name_scope("loss"):
    y_as_float = tf.cast(y, tf.float32)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_as_float,
                                                       logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    y_pred = tf.cast(tf.greater_equal(logits, 0), tf.int32)
    correct = tf.equal(y_pred, y)
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
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

X_test1, y_test1 = generate_batch(X_test, y_test, batch_size=len(X_test))
with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        # if the checkpoint file exists, restore the model and load the epoch
        # number
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = generate_batch(X_train1, y_train1, batch_size)
            acc_train, loss_train, _ = sess.run(
                [accuracy, loss, training_op], feed_dict={X: X_batch, y: y_batch})
        print("epoch:", epoch, "\tTrain accuracy:{:.3f}%".format(
            acc_train * 100), "\tTrain loss:{:.5f}".format(loss_train))
        if epoch % 5 == 0:
            acc_dev, loss_dev = sess.run(
                [accuracy, loss], feed_dict={X: X_test1, y: y_test1})
            print("epoch:", epoch, "\tTest accuracy:{:.3f}%".format(
                acc_train * 100), "\tTest loss:{:.5f}".format(loss_train))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            if loss_dev < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_dev
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break

    os.remove(checkpoint_epoch_path)


# with tf.Session() as sess:
#     saver.restore(sess, final_model_path)
#     acc_test = accuracy.eval(feed_dict={X: X_test1, y: y_test1})
#     print("Final test accuracy: {:.2f}%".format(acc_test * 100))


# tensorboard --logdir=tf_logs


# 注意最后损失函数的使用
