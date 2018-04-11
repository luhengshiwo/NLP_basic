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

X_train1 = X_train[y_train < 5]
X_test1 = X_test[y_test < 5]
y_train1 = y_train[y_train < 5]
y_test1 = y_test[y_test < 5]
X_valid1 = mnist.validation.images[mnist.validation.labels < 5]
y_valid1 = mnist.validation.labels[mnist.validation.labels < 5]

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
final_model_path = "./my_logreg_model"


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
he_init = tf.contrib.layers.variance_scaling_initializer()

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden, name="hidden1", kernel_initializer=he_init,
                              activation=tf.nn.elu)
    hidden2 = tf.layers.dense(hidden1, n_hidden, name="hidden2", kernel_initializer=he_init,
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
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

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
    init.run()
    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train1))
        for rnd_indices in np.array_split(rnd_idx, len(X_train1) // batch_size):
            X_batch, y_batch = X_train1[rnd_indices], y_train1[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train, loss_train, summary_accuracy_train, summary_loss_train = sess.run(
            [accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_train1, y: y_train1})
        # file_writer.add_summary(summary_accuracy_train, epoch)
        # file_writer.add_summary(summary_loss_train, epoch)
        acc_dev, loss_dev, summary_accuracy_dev, summary_loss_dev = sess.run(
            [accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_valid1, y: y_valid1})
        file_writer.add_summary(summary_accuracy_dev, epoch)
        file_writer.add_summary(summary_loss_dev, epoch)
        print("epoch:", epoch, "\tTrain accuracy:{:.3f}%".format(
            acc_train * 100), "\tVal accuracy:{:.3f}%".format(acc_dev * 100))
        print("\tTrain loss:{:.5f}".format(loss_train),
              "\tVal loss:{:.5f}".format(loss_dev))
        if epoch % 5 == 0:
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

    # y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
    os.remove(checkpoint_epoch_path)


# tensorboard --logdir=tf_logs