#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib as tc
from generate import generate_batch
from generate import embed
from generate import id2words
from evaluation_utils import evaluate
import os
# 准备工作，
# 准备两段语料，1，一个中文一个英文，2，准备一个可以batch传输的函数，3，准备两个字典4，准备生成两个词向量
# 对于步奏1，tgt需要生成一个decoder_outputs,并需要生成decoder_length，target_weights
# 对于步奏2，需要将一段话dims的0和1转置，并对所有句子进行max_padding,同时返回句子的长度
# 我觉得这些函数在tensorflow的原生tutorials里面有
s_vec, t_vec, s_voc, t_voc = embed()
tgt_vocab_size = len(t_vec)
batch_size = 32
num_units = 100
max_gradient_norm = 5
learning_rate = 0.01
n_epochs = 10
tgt_sos = '\<s>'
tgt_eos = '\</s>'
tgt_sos_id = t_voc.index(tgt_sos)
tgt_eos_id = t_voc.index(tgt_eos)
pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
source_path = father_path + '/data/cut1000.zh'
target_path = father_path + '/data/small1000.en'
dev_source_path = father_path + '/data/cutdev128.zh'
dev_target_path = father_path + '/data/dev128.en'

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

    # decode
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias=False)
    # projection_layer = tf.python.layers.Dense(tgt_vocab_size,use_bias=False)有区别吗？这边Dense用大写,没有
    # helper
    with tf.variable_scope("decode"):
        train_helper = tc.seq2seq.TrainingHelper(
            decoder_emb_inp, target_sequence_length, time_major=True)
        train_decoder = tc.seq2seq.BasicDecoder(
            decoder_cell, train_helper, encoder_state, output_layer=projection_layer)
        training_outputs, _, _ = tc.seq2seq.dynamic_decode(
            train_decoder, output_time_major=True, swap_memory=True, maximum_iterations=max_target_sequence_length)
    with tf.variable_scope("decode", reuse=True):
        predict_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_decoder,
            tf.fill([batch_size], tgt_sos_id), tgt_eos_id)
        # Decoder
        predict_decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, predict_helper, encoder_state,
            output_layer=projection_layer)
        # Dynamic decoding
        maximum_iterations = tf.round(
            tf.reduce_max(source_sequence_length) * 2)
        predict_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            predict_decoder, output_time_major=True, maximum_iterations=maximum_iterations)

with tf.name_scope('logits'):
    training_logits = tf.identity(training_outputs.rnn_output, 'logits')
    dev_logits_id = tf.identity(training_outputs.sample_id, 'dev_logits')
    dev_out_id = tf.transpose(dev_logits_id, (1, 0))
    predicting_logits = tf.identity(
        predict_outputs.sample_id, name='predictions')
# 这边两步都行
# logits = outputs.rnn_output
with tf.name_scope('loss'):
    masks = tf.sequence_mask(target_sequence_length,
                             max_target_sequence_length, dtype=tf.float32, name='masks')
    # loss
    cost = tf.contrib.seq2seq.sequence_loss(
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
    update_step = optimizer.apply_gradients(zip(clip_gradients, params))

checkpoint = "./trained_model.ckpt"
saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for i in range(int(128 / batch_size)):
            encoder_inputs_batch, decoder_inputs_batch, decoder_outputs_batch, encoder_length_batch, decoder_length_batch = generate_batch(source_path, target_path,
                                                                                                                                           batch_size)
            _, loss = sess.run([update_step, train_loss], feed_dict={encoder_inputs_x: encoder_inputs_batch,
                                                                     decoder_inputs_y1: decoder_inputs_batch, decoder_outputs_y2: decoder_outputs_batch,
                                                                     source_sequence_length: encoder_length_batch, target_sequence_length: decoder_length_batch})
            # print(str(epoch) + ':' + str(loss))
        if epoch % 1 == 0:
            ground_truth_file = 'ground_truth_file'
            trans_file = 'trans_file'
            if os.path.isfile(ground_truth_file):
                os.remove(ground_truth_file)
                os.remove(trans_file)
            f1 = open(ground_truth_file, 'a')
            f2 = open('trans_file', 'a')
            for j in range(int(128 / batch_size)):
                encoder_inputs_dev, decoder_inputs_dev, decoder_outputs_dev, encoder_length_dev, decoder_length_dev = generate_batch(dev_source_path, dev_target_path,
                                                                                                                                     batch_size)
                loss, dev_out = sess.run([train_loss, dev_out_id], feed_dict={encoder_inputs_x: encoder_inputs_dev,
                                                                              decoder_inputs_y1: decoder_inputs_dev, decoder_outputs_y2: decoder_outputs_dev,
                                                                              source_sequence_length: encoder_length_dev, target_sequence_length: decoder_length_dev})
                ground_truth = id2words(decoder_outputs_batch)
                sentences = id2words(dev_out)
                for line1 in ground_truth:
                    f1.write(line1 + '\n')
                for line2 in sentences:
                    f2.write(line2 + '\n')
            f1.close()
            f2.close()
            bleu_score = evaluate(ground_truth_file, trans_file, 'bleu')
            print(bleu_score)
            # print(evaluate)
            print("eval" + str(epoch) + ':' + str(j) + str(loss))

    saver.save(sess, checkpoint)
    print('Model Trained and Saved')


# with tf.Session() as sess:
#     saver.restore(sess, checkpoint)
#     words = [10,22,34]
#     inputs = np.tile(words,(batch_size,1))
#     acc_test = predicting_logits.eval(feed_dict={encoder_inputs_x: inputs,
#                                                  source_sequence_length: [len(words)]*batch_size, target_sequence_length: [len(words)]*batch_size})[0]
#     print(acc_test)
