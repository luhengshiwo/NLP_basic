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

s_vec, t_vec, s_voc, t_voc = embed()
tgt_vocab_size = len(t_vec)
batch_size = 128
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
beam_width = 10
length_penalty_weight = 0.0
num = len(open(source_path).readlines())


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
        #1,attention机制，输入的是转置后的source的encoder
        attention_mechanism = tc.seq2seq.LuongAttention(
            num_units, attention_states, memory_sequence_length=source_sequence_length)
        # 2用一个wrap将attention机制和decoder_cell包起来，作为一个新的cell
        decoder_cell_wrap = tc.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism, attention_layer_size=num_units)
        #3初始化init_state,使用上一步的cell_wrap,和encoder的最后一个state
        init_s = decoder_cell_wrap.zero_state(
            dtype=tf.float32, batch_size=batch_size).clone(cell_state=encoder_state)
        #4helper，输入decode的word_embeddings,长度
        train_helper = tc.seq2seq.TrainingHelper(
            decoder_emb_inp, target_sequence_length, time_major=True)
        #将2，3，4输入到basicdecoder中
        train_decoder = tc.seq2seq.BasicDecoder(
            decoder_cell_wrap, train_helper, init_s, output_layer=projection_layer)
        training_outputs, _, _ = tc.seq2seq.dynamic_decode(
            train_decoder, output_time_major=True, swap_memory=True, maximum_iterations=max_target_sequence_length)

    with tf.variable_scope("decode_outputs", reuse=True):
        #1，start_token复制batch_size次，end_token不动
        start_tokens = tf.fill([batch_size], tgt_sos_id)
        end_token = tgt_eos_id
        #将attention_states(就是源语言的encoder)，源语言长度，复制beam_width次
        beam_attention_states = tf.contrib.seq2seq.tile_batch(
            attention_states, multiplier=beam_width)
        lengths = tf.contrib.seq2seq.tile_batch(
            source_sequence_length, multiplier=beam_width)
        #attebtion机制
        attention_mechanism = tc.seq2seq.LuongAttention(
            num_units, beam_attention_states, memory_sequence_length=lengths)
        #将cell和attention都wrap起来
        decoder_cell_wrap_pre = tc.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                            attention_mechanism=attention_mechanism,
                                                            attention_layer_size=num_units)
        #将encoder的最后一个state复制beam_width次
        init_state = tf.contrib.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)
        #初始化init_state
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
    dev_logits_id = tf.identity(training_outputs.sample_id, 'dev_logits')
    dev_out_id = tf.transpose(dev_logits_id, (1, 0))
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
        for i in range(int(num / batch_size)):
            encoder_inputs_batch, decoder_inputs_batch, decoder_outputs_batch, encoder_length_batch, decoder_length_batch = generate_batch(source_path, target_path,
                                                                                                                                           batch_size)
            _, loss = sess.run([update_step, train_loss], feed_dict={encoder_inputs_x: encoder_inputs_batch,
                                                                     decoder_inputs_y1: decoder_inputs_batch, decoder_outputs_y2: decoder_outputs_batch,
                                                                     source_sequence_length: encoder_length_batch, target_sequence_length: decoder_length_batch})
        if i % 10 == 0:
            print(str(epoch) + ':' + str(loss))
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
#     words = [10, 22, 34]
#     inputs = np.tile(words, (batch_size, 1))
#     acc_test = predict_out_id.eval(feed_dict={encoder_inputs_x: inputs,
#                                                  source_sequence_length: [len(words)] * batch_size, target_sequence_length: [len(words)] * batch_size})[0]

#     print(id2words(acc_test)[0])
