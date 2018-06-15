#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

__author__ = 'luheng'

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib as tc
from generate import generate_batch
from generate import generate_batch_shell
from generate import embeddings
from generate import id2words
from generate import id2words_source
from generate import id2words_target
from evaluation_utils import evaluate
from datetime import datetime
import sys
import os
import time
from util import Progbar
from parameters import Para
import logging
import bleu

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

logger = logging.getLogger("little_try")
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
fh = logging.FileHandler('result.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)


s_vec, t_vec, s_voc, t_voc = embeddings()
tgt_vocab_size = len(t_vec)
batch_size = Para.batch_size
num_units = Para.num_units
max_gradient_norm = Para.max_gradient_norm
learning_rate = Para.learning_rate
n_epochs = Para.n_epochs
tgt_sos = Para.tgt_sos
tgt_eos = Para.tgt_eos
tgt_sos_id = t_voc.index(tgt_sos)
tgt_eos_id = t_voc.index(tgt_eos)
# pwd = os.getcwd()
# father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
beam_width = Para.beam_width
length_penalty_weight = 0.0
train_num = Para.train_num
dev_num = Para.dev_num
epochs_without_progress = 0
max_epochs_without_progress = 50
checkpoint_path = "model/tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "model/my_logreg_model"


# 这边的输入可能需要转置
with tf.name_scope("placeholder"):
    encoder_inputs_x = tf.placeholder(tf.int32, shape=[batch_size, None])
    decoder_inputs_y1 = tf.placeholder(tf.int32, shape=[batch_size, None])
    decoder_outputs_y2 = tf.placeholder(tf.int32, shape=[batch_size, None])
    encoder_inputs = tf.transpose(encoder_inputs_x, (1, 0))
    decoder_inputs = tf.transpose(decoder_inputs_y1, (1, 0))
    decoder_outputs = tf.transpose(decoder_outputs_y2, (1, 0))
    source_sequence_length = tf.placeholder(tf.int32, shape=[batch_size])
    target_sequence_length = tf.placeholder(tf.int32, shape=[batch_size])
    max_target_sequence_length = tf.reduce_max(
        target_sequence_length, name='max_target_len')

with tf.name_scope('wmbeddings'):
    embedding_encoder = tf.Variable(s_vec, trainable=True, name="sourceembed")
    embedding_decoder = tf.Variable(t_vec, trainable=True, name="targetembed")
    # shape = [src_vocab_size,embedding_size]
    encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)
    decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

# encode
with tf.name_scope('encode'):
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_emb_inp, sequence_length=source_sequence_length, time_major=True, dtype=tf.float32)

with tf.name_scope('decode'):
    # decode
    attention_states = tf.transpose(encoder_outputs, (1, 0, 2))
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias=False)
    with tf.variable_scope("decode_outputs"):
        # 1,attention机制，输入的是转置后的source的encoder
        attention_mechanism = tc.seq2seq.LuongAttention(
            num_units, attention_states, memory_sequence_length=source_sequence_length)
        # 2用一个wrap将attention机制和decoder_cell包起来，作为一个新的cell
        decoder_cell_wrap = tc.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism, attention_layer_size=num_units)
        # 3初始化init_state,使用上一步的cell_wrap,和encoder的最后一个state
        init_s = decoder_cell_wrap.zero_state(
            dtype=tf.float32, batch_size=batch_size).clone(cell_state=encoder_state)
        # 4helper，输入decode的word_embeddings,长度
        train_helper = tc.seq2seq.TrainingHelper(
            decoder_emb_inp, target_sequence_length, time_major=True)
        # 将2，3，4输入到basicdecoder中
        train_decoder = tc.seq2seq.BasicDecoder(
            decoder_cell_wrap, train_helper, init_s, output_layer=projection_layer)
        training_outputs, final_state, _ = tc.seq2seq.dynamic_decode(
            train_decoder, output_time_major=True, swap_memory=True, maximum_iterations=max_target_sequence_length)

    with tf.variable_scope("decode_outputs", reuse=True):
        # 1，start_token复制batch_size次，end_token不动
        start_tokens = tf.fill([batch_size], tgt_sos_id)
        end_token = tgt_eos_id
        # 将attention_states(就是源语言的encoder)，源语言长度，复制beam_width次
        beam_attention_states = tc.seq2seq.tile_batch(
            attention_states, multiplier=beam_width)
        lengths = tc.seq2seq.tile_batch(
            source_sequence_length, multiplier=beam_width)
        # attebtion机制
        attention_mechanism = tc.seq2seq.LuongAttention(
            num_units, beam_attention_states, memory_sequence_length=lengths)
        # 将cell和attention都wrap起来
        decoder_cell_wrap_pre = tc.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                            attention_mechanism=attention_mechanism,
                                                            attention_layer_size=num_units)
        # 将encoder的最后一个state复制beam_width次
        init_state = tc.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)
        # 初始化init_state
        decoder_initial_state = decoder_cell_wrap_pre.zero_state(
            dtype=tf.float32, batch_size=batch_size * beam_width).clone(cell_state=init_state)
        # predict_decoder,用beamsearchdecoder整合以上信息
        predict_decoder = tc.seq2seq.BeamSearchDecoder(
            cell=decoder_cell_wrap_pre,
            embedding=embedding_decoder,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,  # initial state of decoder
            beam_width=beam_width,
            output_layer=projection_layer,
            length_penalty_weight=length_penalty_weight)

        # Dynamic decoding
        maximum_iterations = tf.round(
            tf.reduce_max(source_sequence_length) * 2)
        predict_outputs, _, _ = tc.seq2seq.dynamic_decode(
            predict_decoder, output_time_major=True, maximum_iterations=maximum_iterations)

with tf.name_scope('logits'):
    training_logits = tf.identity(training_outputs.rnn_output, 'logits')
    training_logits_id = tf.identity(training_outputs.sample_id, 'dev_logits')
    training_out_id = tf.transpose(training_logits_id, (1, 0))
    predicting_logits = tf.identity(
        predict_outputs.predicted_ids, name='predictions')
    predict_out_id = tf.transpose(predicting_logits, (1, 2, 0))
    # d = tf.shape(predict_out_id)
# 这边两步都行
# logits = outputs.rnn_output
with tf.name_scope('loss'):
    masks = tf.sequence_mask(target_sequence_length,
                             max_target_sequence_length, dtype=tf.float32, name='masks')
    # loss
    cost = tc.seq2seq.sequence_loss(
        training_logits, decoder_outputs, masks)
    train_loss = (tf.reduce_sum(cost) / batch_size)

# calculate and clip gradients
with tf.name_scope('clip_gradients'):
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

# optimization
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # update_step = optimizer.minimize(train_loss)
    update_step = optimizer.apply_gradients(zip(clip_gradients, params))


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
    best_bleu = 0.0
    best_epoch = 0
    with tf.Session() as sess:
        sess.run(init)
        train_gen = generate_batch(train_data, batch_size=batch_size)
        dev_gen = generate_batch(dev_data, batch_size=batch_size)
        for epoch in range(n_epochs):
            prog = Progbar(target=1 + int(train_num / batch_size))
            for i in range(int(train_num / batch_size)):
                encoder_inputs_batch, decoder_inputs_batch, decoder_outputs_batch, encoder_length_batch, decoder_length_batch = next(
                    train_gen)
                _, loss, train_out = sess.run([update_step, train_loss, training_out_id], feed_dict={encoder_inputs_x: encoder_inputs_batch,
                                                                                                     decoder_inputs_y1: decoder_inputs_batch, decoder_outputs_y2: decoder_outputs_batch,
                                                                                                     source_sequence_length: encoder_length_batch, target_sequence_length: decoder_length_batch})
                train_bleu_score, _, _, _, _, _ = bleu.compute_bleu(
                    id2words_source(decoder_inputs_batch), id2words_target(train_out), 4, smooth=False)
                prog.update(i + 1, [("train loss", loss),
                                    ("train bleu:", train_bleu_score)])
            print('\nresult of this train epoch:')
            logger.info("epoch:{}".format(epoch) +
                        "\tepoch_loss:{:.5f}".format(loss))
            if epoch % 1 == 0:
                dev_loss = 0.0
                reference_corpus = []
                translation_corpus = []
                num = int(dev_num / batch_size)
                for i in range(num):
                    encoder_inputs_dev, decoder_inputs_dev, decoder_outputs_dev, encoder_length_dev, decoder_length_dev = next(
                        dev_gen)
                    loss, dev_out = sess.run([train_loss, predict_out_id], feed_dict={encoder_inputs_x: encoder_inputs_dev,
                                                                                      decoder_inputs_y1: decoder_inputs_dev, decoder_outputs_y2: decoder_outputs_dev,
                                                                                      source_sequence_length: encoder_length_dev, target_sequence_length: decoder_length_dev})
                    dev_out_no1 = dev_out[:, 0, :]
                    reference_corpus.extend(
                        id2words_source(decoder_inputs_dev))
                    translation_corpus.extend(id2words_target(dev_out_no1))
                    dev_loss += loss
                bleu_score, _, _, _, _, _ = bleu.compute_bleu(
                    reference_corpus, translation_corpus, 4, smooth=False)
                logger.info("epoch:{}".format(epoch)+"\tDev loss:{:.5f}".format(dev_loss / num)+"\tbleu_dev:{:.5f}".format(bleu_score*100))
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))
                if bleu_score >= best_bleu:
                    logger.info('save epoch:{}'.format(epoch))
                    saver.save(sess, final_model_path)
                    best_bleu = bleu_score
                    best_epoch = epoch
                else:
                    epochs_without_progress += 1
                    if epochs_without_progress > max_epochs_without_progress:
                        logger.info("Early stopping")
                        break
        os.remove(checkpoint_epoch_path)
        tok = time.time()
        cost = tok-tic
        logger.info("best_epoch:{}".format(best_epoch)+"\tbest_bleu:{:.5f}".format(best_bleu))
        logger.info('final training time:{:.2f}'.format(cost))


def do_evaluate(arg):
    test_data = arg.test_data
    ground_truth_file = arg.ground_truth
    trans_file = arg.trans_file
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        test_gen = generate_batch(test_data, batch_size=batch_size)
        # ground_truth_file = 'ground_truth_file'
        # trans_file = 'trans_file'
        if os.path.isfile(ground_truth_file):
            os.remove(ground_truth_file)
            os.remove(trans_file)
        f1 = open(ground_truth_file, 'a')
        f2 = open(trans_file, 'a')
        for j in range(int(128 / batch_size)):
            encoder_inputs_test, decoder_inputs_test, _, encoder_length_test, _ = next(test_gen)
            test_out = sess.run(predict_out_id, feed_dict={
                                     encoder_inputs_x: encoder_inputs_test, source_sequence_length: encoder_length_test})
            test_out_no1 = test_out[:, 0, :]
            ground_truth = id2words(decoder_inputs_test)
            sentences = id2words(test_out_no1)
            for line1 in ground_truth:
                f1.write(line1 + '\n')
            for line2 in sentences:
                f2.write(line2 + '\n')
        f1.close()
        f2.close()
        bleu_score = evaluate(ground_truth_file, trans_file, 'bleu')
        logger.info('evaluation done! out_path:{}'.format(trans_file)+"\tbleu_score:{:.5f}".format(bleu_score))
        # print(bleu_score)

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
            else:
                shell_gen = generate_batch_shell(sentence,batch_size)
                source, source_seq_length = next(
                    shell_gen)
                preds = sess.run(predict_out_id, feed_dict={encoder_inputs_x: source, source_sequence_length: source_seq_length})
                predict = preds[:, 0, :]
                trans_sentence = id2words(predict)[0]
                logger.info("输入原句：{}".format(sentence))
                logger.info("输出结果：{}".format(trans_sentence))
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
        '-g', '--ground-truth', type=str, default="data/ground_truth", help="Ground data")
    command_parser.add_argument(
        '-tr', '--trans-file', type=str, default="data/trans_file", help="Trans data")
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('test', help=' ')
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
