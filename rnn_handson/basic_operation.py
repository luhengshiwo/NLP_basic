#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib as tc


# peephole
# 问题，只有lstm_cell有peephole的功能吗？
lstm_cell = tc.rnn.LSTMCell(num_units=n_neurons, use_peepholes=True)
lstm_cell = tc.rnn.BasicLSTMCell(num_units=n_neurons)


# cell
basic_cell = tc.rnn.BasicRNNCell(num_units=n_neurons)
basic_cell = tc.rnn.BasicLSTMCell(num_units=n_neurons)
basic_cell = tc.rnn.GRUCell(num_units=n_neurons)


# dynamic_rnn
rnn_outputs, states = tf.nn.dynamic_rnn(cell, inputs=x, sequence_length=None, initial_state=None,
                                        dtype=None, parallel_iterations=None, swap_memory=False,
                                        time_major=False, scope=None)
# 如果cell 为tc.rnn.BasicLSTMCell(num_units=n_neurons)，那么，states是一个tuple，
# initial_state,为初始化的state，可以自定义,问题，如果是LSTM或者GRU，那么自定义的initial_state里面需要是什么格式的？
# time_major是看要不要将数据的第0维和第1维转置
# states里面是什么东西？

# bi_rnn
f_cell = tc.rnn.BasicLSTMCell(num_units=n_neurons)
b_cell = tc.rnn.BasicLSTMCell(num_units=n_neurons)
outputs, output_states = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, inputs, sequence_length=None, initial_state_fw=None,
                                                         initial_state_bw=None, dtype=None, parallel_iterations=None,
                                                          swap_memory=False, time_major=False, scope=None)
# 1. outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。假设
# time_major=false,tensor的shape为[batch_size, max_time, depth]。可以使用tf.concat(outputs, 2)将其拼接。
# 2. output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组。
# 如果cell使用的是lstmcell,output_state_fw和output_state_bw的类型为LSTMStateTuple。
# LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。

#multi-layer
n_neurons = 100
n_layers = 3
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                      activation=tf.nn.relu)
          for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers,state_is_tuple=False)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
#如果是rnn或者gru，states就是h,如果是lstm，states是一个tuple，包含(c,h)

#dropout

#普通rnn，用DropoutWrapper将cell包起来,将keep_prob做为placeholer，
#并用placeholder_with_default代替普通的placeholder
#这样在传值的时候，只有在训练的时候要传keep_prob，其他时候不需要传值
keep_prob = tf.placeholder_with_default(1.0, shape=())
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
rnn_outputs, states = tf.nn.dynamic_rnn(cell_drop, X, dtype=tf.float32)

n_iterations = 1500
batch_size = 50
train_keep_prob = 0.5

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        _, mse = sess.run([training_op, loss],
                          feed_dict={X: X_batch, y: y_batch,
                                     keep_prob: train_keep_prob})


#看multi-layer
keep_prob = tf.placeholder_with_default(1.0, shape=())
cells = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
         for layer in range(n_layers)]
cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
              for cell in cells]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)



#Attention




#word embedding