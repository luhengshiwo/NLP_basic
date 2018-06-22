#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

__author__ = 'luheng'

import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from batch_data import generate_batch
from batch_data import generate_batch_shell
from batch_data import generate_batch_evaluate
from batch_data import embeddings
from datetime import datetime
from util import Progbar
import sys
from sklearn import metrics
import os
import time
from parameters import Para
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

logger = logging.getLogger("little_try")
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
fh = logging.FileHandler('result.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

vec, voc = embeddings()
vocab_size = len(vec)
batch_size = Para.batch_size
num_units = Para.num_units
max_gradient_norm = Para.max_gradient_norm
learning_rate = Para.learning_rate
n_epochs = Para.n_epochs
n_outputs = Para.n_outputs
train_keep_prob = Para.train_keep_prob
train_num = Para.train_num
dev_num = Para.dev_num
test_num = Para.test_num
l2_rate = Para.l2_rate
best_loss = np.infty
threshold = Para.threshold
epochs_without_progress = 0
max_epochs_without_progress = 50
checkpoint_path = "model/tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "model/my_logreg_model"

'''
tricks:
1,he_init or Xavier or None
2,GRU or Bi-GRU
3,LuongAttention or BahdanauAttention
4,L1_regularization or L2_regularization or None
5,Gradient Clipping or None
6,Adam Momentum RMSProp Adadelta
'''
with tf.name_scope("placeholder"):
    encode_question = tf.placeholder(tf.int32, shape=[batch_size, None])
    encode_answer = tf.placeholder(tf.int32, shape=[batch_size, None])
    question = tf.transpose(encode_question, (1, 0))
    answer = tf.transpose(encode_answer, (1, 0))
    question_sequence_length = tf.placeholder(tf.int32, shape=[batch_size])
    answer_sequence_length = tf.placeholder(tf.int32, shape=[batch_size])
    simi_values = tf.placeholder(tf.int32, shape=[batch_size])
    keep_prob = tf.placeholder_with_default(1.0, shape=())
    max_target_sequence_length = tf.reduce_max(
        answer_sequence_length, name='max_target_len')

with tf.name_scope('word_embeddings'):
    embeddings = tf.Variable(vec, trainable=True, name="embeds")
    question_emb = tf.nn.embedding_lookup(embeddings, question)
    answer_emb = tf.nn.embedding_lookup(embeddings, answer)

with tf.name_scope("decode"):
    # initializer
    he_init = tc.layers.variance_scaling_initializer()
    Xavier = tc.layers.xavier_initializer()

    '''
    GRU
    '''
    # with tf.variable_scope("question"):
    #     question_cell = tf.nn.rnn_cell.GRUCell(num_units)
    #     question_drop = tc.rnn.DropoutWrapper(
    #         question_cell, input_keep_prob=keep_prob)
    #     question_outputs, question_state = tf.nn.dynamic_rnn(
    #         question_drop, question_emb, sequence_length=question_sequence_length, time_major=True, dtype=tf.float32)
    '''
    Bi-GRU
    '''
    with tf.variable_scope("question"):
        forward_cell = tc.rnn.GRUCell(num_units, kernel_initializer=he_init)
        forward_cell_drop = tc.rnn.DropoutWrapper(
            forward_cell, input_keep_prob=keep_prob)
        backward_cell = tc.rnn.GRUCell(num_units, kernel_initializer=he_init)
        backward_cell_drop = tc.rnn.DropoutWrapper(
            backward_cell, input_keep_prob=keep_prob)
        bi_question_outputs, bi_question_state = tf.nn.bidirectional_dynamic_rnn(
            forward_cell_drop, backward_cell_drop, question_emb, sequence_length=question_sequence_length, time_major=True, dtype=tf.float32)
        question_state = tf.add_n(bi_question_state)
        question_outputs = tf.add_n(bi_question_outputs)

    with tf.variable_scope("answer"):
        attention_states = tf.transpose(question_outputs, (1, 0, 2))
        answer_cell = tc.rnn.GRUCell(num_units, kernel_initializer=he_init)
        answer_drop = tc.rnn.DropoutWrapper(
            answer_cell, input_keep_prob=keep_prob)
        """
        LuongAttention
        """
        attention_mechanism = tc.seq2seq.LuongAttention(
            num_units, attention_states, memory_sequence_length=question_sequence_length)
        decoder_cell_wrap = tc.seq2seq.AttentionWrapper(
            answer_drop, attention_mechanism, attention_layer_size=num_units)
        """
        BahdanauAttention
        """
        # attention_mechanism = tc.seq2seq.BahdanauAttention(
        #     num_units, attention_states, memory_sequence_length=question_sequence_length)
        # decoder_cell_wrap = tc.seq2seq.AttentionWrapper(
        #     answer_drop, attention_mechanism, attention_layer_size=num_units)
        init_s = decoder_cell_wrap.zero_state(
            dtype=tf.float32, batch_size=batch_size).clone(cell_state=question_state)
        # projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias=False)
        train_helper = tc.seq2seq.TrainingHelper(
            answer_emb, answer_sequence_length, time_major=True)
        train_decoder = tc.seq2seq.BasicDecoder(
            decoder_cell_wrap, train_helper, init_s, output_layer=None)
        training_outputs, answer_state, _ = tc.seq2seq.dynamic_decode(
            train_decoder, output_time_major=True, swap_memory=True, maximum_iterations=max_target_sequence_length)

with tf.name_scope("logits"):
    logits = tf.layers.dense(answer_state.cell_state, n_outputs)
    labels = simi_values
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    base_loss = tf.reduce_mean(xentropy)
    # regularization
    reg_losses = tc.layers.apply_regularization(
        tc.layers.l2_regularizer(l2_rate), tf.trainable_variables())
    loss = base_loss + reg_losses
    loss_summary = tf.summary.scalar('log_loss', loss)

with tf.name_scope("optimizer"):
    # optimizer = tf.train.MomentumOptimizer(momentum=0.9,learning_rate=learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # training_op = optimizer.minimize(loss)
    # Gradient Clipping
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
                  for grad, var in grads_and_vars]
    training_op = optimizer.apply_gradients(capped_gvs)
    pred = tf.argmax(logits, axis=-1)
    correct = tf.nn.in_top_k(logits, labels, 1)
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


def do_train(args):
    tic = time.time()
    train_data = args.train_data
    dev_data = args.dev_data
    epochs_without_progress = 0
    max_epochs_without_progress = 50
    best_f1 = 0.0
    best_epoch = 0
    best_precision = 0.0
    best_recall = 0.0
    with tf.Session(config=config) as sess:
        train_gen = generate_batch(
            train_data, batch_size=batch_size)
        dev_gen = generate_batch(dev_data, batch_size=batch_size)
        if os.path.isfile(checkpoint_epoch_path):
            with open(checkpoint_epoch_path, "rb") as f:
                start_epoch = int(f.read())
            logger.info(
                "Training was interrupted. Continuing at epoch：{}".format(start_epoch))
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
                    encode_question: source_batch_pad, encode_answer: target_batch_pad,
                    simi_values: simi_input_batch, question_sequence_length: source_seq_length, answer_sequence_length: target_seq_length, keep_prob: train_keep_prob})
                y_true = np.array(simi_input_batch)
                precision = metrics.precision_score(y_true, pred_,average='micro')
                recall = metrics.recall_score(y_true, pred_,average='micro')
                f1 = metrics.f1_score(y_true, pred_,average='micro')
                if i % 10 == 0:
                    file_writer.add_summary(
                        loss_summary_, epoch * train_num + i)
                prog.update(i + 1, [("train loss", loss_), ("precision",
                                                            precision), ("recall", recall), ("f1", f1)])
            print('\nresult of this train epoch:')
            logger.info("epoch:{}".format(epoch) +
                        "\tepoch_loss:{:.5f}".format(loss_))
            if epoch % 1 == 0:
                num = int(dev_num / batch_size)
                # dev_accuracy = 0.0
                dev_loss = 0.0
                y_true = []
                y_pred = []
                for i in range(num):
                    dev1_pad, dev1_seq_length, dev2_pad, dev2_seq_length, dev_simi_batch = next(
                        dev_gen)
                    pred_dev, loss_dev = sess.run([pred, loss], feed_dict={
                        encode_question: dev1_pad, encode_answer: dev2_pad,
                        simi_values: dev_simi_batch, question_sequence_length: dev1_seq_length, answer_sequence_length: dev2_seq_length})
                    y_true = np.append(y_true, dev_simi_batch)
                    y_pred = np.append(y_pred, pred_dev)
                    dev_loss += loss_dev
                y_true = np.array(y_true)
                precision_dev = metrics.precision_score(y_true, y_pred,average='micro')
                recall_dev = metrics.recall_score(y_true, y_pred,average='micro')
                f1_dev = metrics.f1_score(y_true, y_pred,average='micro')
                logger.info("epoch:{}".format(epoch)+"\tDev loss:{:.5f}".format(dev_loss / num)+"\tprecision_dev:{:.5f}".format(
                    precision_dev)+"\trecall_dev:{:.5f}".format(recall_dev)+"\tf1_dev:{:.5f}".format(f1_dev))
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))
                if f1_dev > best_f1:
                    logger.info('save epoch:{}'.format(epoch))
                    saver.save(sess, final_model_path)
                    best_f1 = f1_dev
                    best_precision = precision_dev
                    best_epoch = epoch
                    best_recall = recall_dev
                else:
                    epochs_without_progress += 1
                    if epochs_without_progress > max_epochs_without_progress:
                        logger.info("Early stopping")
                        break
        os.remove(checkpoint_epoch_path)
        tok = time.time()
        cost = tok-tic
        logger.info("best_epoch:{}".format(best_epoch)+"\tbest_precision:{:.5f}".format(
                    best_precision)+"\tbest_recall:{:.5f}".format(best_recall)+"\tbest_f1:{:.5f}".format(best_f1))
        logger.info('final training time:{:.2f}'.format(cost))


def do_evaluate(args):
    tic = time.time()
    inpath = args.test_data
    outpath = args.output_data
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        test_gen = generate_batch_evaluate(inpath)
        with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
            for _ in fin:
                ids, source, source_seq_length, target, target_seq_length = next(
                    test_gen)
                preds = sess.run(pred, feed_dict={encode_question: source, encode_answer: target,
                                                  question_sequence_length: source_seq_length, answer_sequence_length: target_seq_length})
                predict = preds[0]
                fout.write(ids[0] + '\t' + str(predict[0]) + '\n')
    logger.info('evaluation done! out_path:{}'.format(outpath))
    tok = time.time()
    cost = tok-tic
    single_cost = cost*1000/(test_num)
    logger.info(
        'each evaluate time of single data:{:.2f}ms'.format(single_cost))
# # tensorboard --logdir=tf_logs


def do_shell(_):
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        sentence = '请输入句子,要退出请输入：bye'
        logger.info(sentence)
        while sentence != 'bye':
            sentence = input("input> ")
            if sentence == 'bye':
                logger.info("准备退出...")
            elif sentence == '':
                logger.info("请不要打回车玩哦！")
            elif '__user__' not in sentence:
                logger.info("请输入正确的格式！提示：Question__user__Answer;退出请输入：bye")
            else:
                shell_gen = generate_batch_shell(sentence)
                source, source_seq_length, target, target_seq_length = next(
                    shell_gen)
                preds = sess.run(pred, feed_dict={encode_question: source, encode_answer: target,
                                                  question_sequence_length: source_seq_length, answer_sequence_length: target_seq_length})
                predict = preds[0]
                logger.info("输入原句：{}".format(sentence))
                if predict[0] == 1:
                    logger.info("输出结果：该句识别准确")
                else:
                    logger.info("输出结果：该句识别不准确")
        logger.info('谢谢使用，再见！')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains and tests an classification model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help=' ')
    command_parser.add_argument(
        '-td', '--train-data', type=str, default="data/train.csv", help="Training data")
    command_parser.add_argument(
        '-dd', '--dev-data', type=str, default="data/dev.csv", help="Dev data")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help=' ')
    command_parser.add_argument(
        '-t', '--test-data', type=str, default="data/test.csv", help="Evaluate data")
    command_parser.add_argument(
        '-o', '--output-data', type=str, default="data/out.csv", help="Output data")
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('test', help=' ')
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
