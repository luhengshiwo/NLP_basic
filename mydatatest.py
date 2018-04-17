#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



a = tf.constant(1.0,name='a')

graph = tf.Graph()
with graph.as_default():
    x1 = tf.Variable(1)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        b = sess.run(x1)
        print(b)

with tf.Session() as sess:
    b = sess.run(a)
    print(b)
with tf.Session() as sess:
    b = sess.run(a)
    print(b)

with graph.as_default():
    x2 = tf.Variable(2.0)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        b = sess.run(x2)
        print(b)


