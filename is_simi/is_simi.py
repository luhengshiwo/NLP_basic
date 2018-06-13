#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from batch_data import generate_batch
from batch_data import embeddings
from datetime import datetime
from util import Progbar
import sys
from sklearn import metrics
import os
import time
from config import Config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

vec, voc = embeddings()
vocab_size = len(vec)
batch_size = Config.batch_size
# batch_size = 1
num_units = Config.num_units
max_gradient_norm = Config.max_gradient_norm
learning_rate = Config.learning_rate
n_epochs = Config.n_epochs
n_outputs = Config.n_outputs
train_keep_prob = Config.train_keep_prob
train_num = Config.train_num
dev_num = Config.dev_num
best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50
checkpoint_path = "model/tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "model/my_logreg_model"


with tf.name_scope("placeholder"):
    encode_x1 = tf.placeholder(tf.int32, shape=[batch_size, None])
    encode_x2 = tf.placeholder(tf.int32, shape=[batch_size, None])
    x1 = tf.transpose(encode_x1, (1, 0))
    x2 = tf.transpose(encode_x2, (1, 0))
    x1_sequence_length = tf.placeholder(tf.int32, shape=[batch_size])
    x2_sequence_length = tf.placeholder(tf.int32, shape=[batch_size])
    simi_values = tf.placeholder(tf.int32, shape=[batch_size])
    keep_prob = tf.placeholder_with_default(1.0, shape=())

with tf.name_scope('wmbeddings'):
    embeddings = tf.Variable(vec, trainable=True, name="embeds")
    x1_emb = tf.nn.embedding_lookup(embeddings, x1)
    x2_emb = tf.nn.embedding_lookup(embeddings, x2)
with tf.name_scope("decode"):
    with tf.variable_scope("x1"):
        x1_f_cell = tf.nn.rnn_cell.GRUCell(num_units)
        x1_b_cell = tf.nn.rnn_cell.GRUCell(num_units)
        x1_f_drop = tc.rnn.DropoutWrapper(
            x1_f_cell, input_keep_prob=keep_prob)
        x1_b_drop = tc.rnn.DropoutWrapper(
            x1_b_cell, input_keep_prob=keep_prob)
        x1_outputs, x1_state = tf.nn.bidirectional_dynamic_rnn(
            x1_f_drop, x1_b_drop, x1_emb, sequence_length=x1_sequence_length, time_major=True, dtype=tf.float32)
    with tf.variable_scope("x2"):
        x2_f_cell = tf.nn.rnn_cell.GRUCell(num_units)
        x2_b_cell = tf.nn.rnn_cell.GRUCell(num_units)
        x2_f_drop = tc.rnn.DropoutWrapper(
            x2_f_cell, input_keep_prob=keep_prob)
        x2_b_drop = tc.rnn.DropoutWrapper(
            x2_b_cell, input_keep_prob=keep_prob)
        x2_outputs, x2_state = tf.nn.bidirectional_dynamic_rnn(
            x2_f_drop, x2_b_drop, x2_emb, sequence_length=x2_sequence_length, time_major=True, dtype=tf.float32)
with tf.name_scope("logits"):
    x1_f_state, x1_b_state = x1_state
    x2_f_state, x2_b_state = x2_state
    x1_state_new = tf.concat([x1_f_state, x1_b_state], axis=-1)
    x2_state_new = tf.concat([x2_f_state, x2_b_state], axis=-1)
    states = tf.concat([x1_state_new, x2_state_new], axis=-1)
    logits = tf.layers.dense(states, n_outputs)
    logits = tf.layers.dropout(logits,rate=1-keep_prob)
    labels = tf.reshape(simi_values, (batch_size, 1))
    labels = tf.cast(labels, tf.float32)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(xentropy)
    loss_summary = tf.summary.scalar('log_loss', loss)
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    logits_soft = tf.sigmoid(logits)
    pred = tf.cast(tf.round(logits_soft), tf.int32)
    correct = tf.equal(pred, simi_values)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('log_accuracy', accuracy)


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "model/tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


logdir = log_dir("logreg")

init = tf.global_variables_initializer()
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())



best_f1 = 0.0
with tf.Session(config=config) as sess:
    train_gen = generate_batch(
        'data/train.csv', batch_size=batch_size)
    dev_gen = generate_batch('data/dev.csv', batch_size=batch_size)
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)
    for epoch in range(n_epochs):
        prog = Progbar(target=1 + int(train_num / batch_size))
        for i in range(int(train_num / batch_size)):
            source_batch_pad, source_seq_length, target_batch_pad, target_seq_length, simi_input_batch = next(
                train_gen)
            pred_, logits_, loss_, loss_summary_, _ = sess.run([pred, logits, loss, loss_summary, training_op], feed_dict={
                encode_x1: source_batch_pad, encode_x2: target_batch_pad,
                simi_values: simi_input_batch, x1_sequence_length: source_seq_length, x2_sequence_length: target_seq_length, keep_prob: train_keep_prob})
            y_true = np.array(simi_input_batch)
            precision = metrics.precision_score(y_true, pred_)
            recall = metrics.recall_score(y_true, pred_)
            f1 = metrics.f1_score(y_true, pred_)
            if i % 10 == 0:
                file_writer.add_summary(
                    loss_summary_, epoch * train_num + i)
            prog.update(i + 1, [("train loss", loss_), ("precision",
                                                        precision), ("recall", recall), ("f1", f1)])
        print("epoch:{}".format(epoch), "epoch_loss:{:.5f}".format(loss_))
        if epoch % 1 == 0:
            num = int(dev_num / batch_size)
            dev_accuracy = 0.0
            dev_loss = 0.0
            y_true = []
            y_pred = []
            for i in range(num):
                dev1_pad, dev1_seq_length, dev2_pad, dev2_seq_length, dev_simi_batch = next(
                    dev_gen)
                pred_dev, loss_dev = sess.run([pred, loss], feed_dict={
                    encode_x1: dev1_pad, encode_x2: dev2_pad,
                    simi_values: dev_simi_batch, x1_sequence_length: dev1_seq_length, x2_sequence_length: dev2_seq_length})
                y_true = np.append(y_true, dev_simi_batch)
                y_pred = np.append(y_pred, pred_dev)
                dev_loss += loss_dev
            y_true = np.array(y_true)
            precision_dev = metrics.precision_score(y_true, y_pred)
            recall_dev = metrics.recall_score(y_true, y_pred)
            f1_dev = metrics.f1_score(y_true, y_pred)
            print("epoch:{}".format(epoch), "\tDev loss:{:.5f}".format(dev_loss / num), "\tprecision_dev:{:.5f}".format(precision_dev),
                    "\trecall_dev:{:.5f}".format(recall_dev), "\tf1_dev:{:.5f}".format(f1_dev))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            if f1_dev > best_f1:
                print('save epoch:{}'.format(epoch))
                saver.save(sess, final_model_path)
                best_f1 = f1_dev
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break

    os.remove(checkpoint_epoch_path)


def finaluse(inpath, outpath):
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        test_gen = generate_batch(
            inpath, is_train=False, batch_size=batch_size)
        with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
            for line in fin:
                ids, source, source_seq_length, target, target_seq_length = next(
                    test_gen)
                preds = sess.run(pred, feed_dict={encode_x1: source, encode_x2: target,
                                                  x1_sequence_length: source_seq_length, x2_sequence_length: target_seq_length})
                predict = preds[0]
                fout.write(ids[0] + '\t' + str(predict[0]) + '\n')

# # tensorboard --logdir=tf_logs


# if __name__ == '__main__':
#     # inputpath = 'data/test.csv'
#     # outpath = 'data/out.csv'
#     # # finaluse(sys.argv[1], sys.argv[2])
#     # finaluse(inputpath, outpath)
#     training()
