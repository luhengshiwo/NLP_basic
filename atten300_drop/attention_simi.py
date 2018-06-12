#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from generate import generate_batch
from generate import embed
from generate import id2words
from generate import evaluate
import os
from datetime import datetime
from util import Progbar
import sys
from sklearn import metrics
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

vec, voc = embed()
vocab_size = len(vec)
batch_size = 512
num_units = 100
max_gradient_norm = 5
learning_rate = 0.001
n_epochs = 10
n_outputs = 1
train_keep_prob = 0.6
pwd = os.getcwd()
train1 = pwd + '/data/train1.conll'
train2 = pwd + '/data/train2.conll'
train_simi = pwd + '/data/train_simi.conll'
dev1 = pwd + '/data/dev1.conll'
dev2 = pwd + '/data/dev2.conll'
dev_simi = pwd + '/data/dev_simi.conll'
best_loss = np.infty
best_f1 = 0.0
epochs_without_progress = 0
max_epochs_without_progress = 50
checkpoint_path = "./tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_logreg_model"
tgt_sos = '\<s>'
tgt_eos = '\</s>'
tgt_sos_id = voc.index(tgt_sos)
tgt_eos_id = voc.index(tgt_eos)

with tf.name_scope("placeholder"):
    encode_x1 = tf.placeholder(tf.int32, shape=[batch_size, None])
    encode_x2 = tf.placeholder(tf.int32, shape=[batch_size, None])
    x1 = tf.transpose(encode_x1, (1, 0))
    x2 = tf.transpose(encode_x2, (1, 0))
    x1_sequence_length = tf.placeholder(tf.int32, shape=[batch_size])
    x2_sequence_length = tf.placeholder(tf.int32, shape=[batch_size])
    simi_values = tf.placeholder(tf.int32, shape=[batch_size])
    keep_prob = tf.placeholder_with_default(1.0, shape=())
    max_target_sequence_length = tf.reduce_max(
        x2_sequence_length, name='max_target_len')
with tf.name_scope('wmbeddings'):
    embeddings = tf.Variable(vec, trainable=True, name="embeds")
    x1_emb = tf.nn.embedding_lookup(embeddings, x1)
    x2_emb = tf.nn.embedding_lookup(embeddings, x2)
with tf.name_scope("decode"):
    with tf.variable_scope("x1"):
        x1_cell = tf.nn.rnn_cell.GRUCell(num_units)
        x1_drop = tf.contrib.rnn.DropoutWrapper(
            x1_cell, input_keep_prob=keep_prob)
        x1_outputs, x1_state = tf.nn.dynamic_rnn(
            x1_drop, x1_emb, sequence_length=x1_sequence_length, time_major=True, dtype=tf.float32)
    with tf.variable_scope("x2"):
        x2_cell = tf.nn.rnn_cell.GRUCell(num_units)
        x2_drop = tf.contrib.rnn.DropoutWrapper(
            x2_cell, input_keep_prob=keep_prob)
        # projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias=False)
        train_helper = tc.seq2seq.TrainingHelper(
            x2_emb, x2_sequence_length, time_major=True)
        train_decoder = tc.seq2seq.BasicDecoder(
            x2_cell, train_helper, x1_state, output_layer=None)
        _, x2_state, _ = tc.seq2seq.dynamic_decode(
            train_decoder, output_time_major=True, swap_memory=True, maximum_iterations=max_target_sequence_length)

with tf.name_scope("logits"):
    logits = tf.layers.dense(x2_state, n_outputs)
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
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


logdir = log_dir("logreg")

init = tf.global_variables_initializer()
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session(config=config) as sess:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)
    for epoch in range(n_epochs):
        train_num = 39346
        # train_num = 32
        prog = Progbar(target=1 + int(train_num / batch_size))
        for i in range(int(train_num / batch_size)):
            source_batch_pad, target_batch_pad, simi_input_batch, source_seq_length, target_seq_length = generate_batch(
                train1, train2, train_simi, batch_size)
            pred_, logits_, loss_, loss_summary_, _ = sess.run([pred, logits, loss, loss_summary, training_op], feed_dict={
                encode_x1: source_batch_pad, encode_x2: target_batch_pad,
                simi_values: simi_input_batch, x1_sequence_length: source_seq_length, x2_sequence_length: target_seq_length, keep_prob: train_keep_prob})
            y_true = np.array(simi_input_batch)
            precision = metrics.precision_score(y_true, pred_)
            recall = metrics.recall_score(y_true, pred_)
            f1 = metrics.f1_score(y_true, pred_)
            if i % 10 == 0:
                file_writer.add_summary(loss_summary_, epoch * train_num + i)
            prog.update(i + 1, [("train loss", loss_), ("precision",
                                                        precision), ("recall", recall), ("f1", f1)])
        print("epoch:{}".format(epoch), "epoch_loss:{:.5f}".format(loss_))
        if epoch % 1 == 0:
            dev_num = 1984
            # dev_num = 32
            num = int(dev_num / batch_size)
            dev_accuracy = 0.0
            dev_loss = 0.0
            y_true = []
            y_pred = []
            for i in range(num):
                dev1_pad, dev2_pad, dev_simi_batch, dev1_seq_length, dev2_seq_length = generate_batch(
                    dev1, dev2, dev_simi, batch_size)
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
                print("epoch:{}".format(epoch))
                saver.save(sess, final_model_path)
                best_f1 = f1_dev
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break

    os.remove(checkpoint_epoch_path)


# def finaluse(inpath, outpath):
#     with tf.Session() as sess:
#         saver.restore(sess, checkpoint_path)
#         with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
#             for line in fin:
#                 ids, source, target, source_seq_length, target_seq_length = evaluate(
#                     line,batch_size)
#                 preds = sess.run(pred, feed_dict={encode_x1: source, encode_x2: target,
#                                                  x1_sequence_length: source_seq_length, x2_sequence_length: target_seq_length})
#                 predict = preds[0][0]
#                 fout.write(ids + '\t'+str(predict)+'\n')

# # tensorboard --logdir=tf_logs

# if __name__ == '__main__':
#     # inputpath = pwd + '/data/atec_nlp_sim_train.csv'
#     # outpath = pwd + '/data/out.csv'
#     finaluse(sys.argv[1], sys.argv[2])
#     # finaluse(inputpath,outpath)
