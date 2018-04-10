#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import os
PROJECT_ROOT_DIR = ".."
CHAPTER_ID = "deep"
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


#Xavier and He initialization
# n_inputs = 28 * 28  # MNIST
# n_hidden1 = 300

# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
# he_init = tf.contrib.layers.variance_scaling_initializer()
# Xavier = tf.contrib.layers.xavier_initializer()
# hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
#                           kernel_initializer=he_init, name="hidden1")
# hidden2 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
#                           kernel_initializer=Xavier, name="hidden2")

z = np.linspace(-5, 5, 200)

#vanishing
# def logit(z):
#     return 1 / (1 + np.exp(-z))

# plt.plot([-5, 5], [0, 0], 'k-')
# plt.plot([-5, 5], [1, 1], 'k--')
# plt.plot([0, 0], [-0.2, 1.2], 'k-')
# plt.plot([-5, 5], [-3/4, 7/4], 'g--')
# plt.plot(z, logit(z), "b-", linewidth=2)
# props = dict(facecolor='black', shrink=0.1)
# plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
# plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
# plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")
# plt.grid(True)
# plt.title("Sigmoid activation function", fontsize=14)
# plt.axis([-5, 5, -0.2, 1.2])

# save_fig("sigmoid_saturation_plot")
# plt.show()


#leaky_relu
# def leaky_relu(z, alpha=0.01):
#     return np.maximum(alpha*z, z)

# plt.plot(z, leaky_relu(z, 0.05), "b-", linewidth=2)
# plt.plot([-5, 5], [0, 0], 'k-')
# plt.plot([0, 0], [-0.5, 4.2], 'k-')
# plt.grid(True)
# props = dict(facecolor='black', shrink=0.1)
# plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
# plt.title("Leaky ReLU activation function", fontsize=14)
# plt.axis([-5, 5, -0.5, 4.2])

# save_fig("leaky_relu_plot")
# plt.show()

#elu
def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)
# plt.plot(z, elu(z), "b-", linewidth=2)
# plt.plot([-5, 5], [0, 0], 'k-')
# plt.plot([-5, 5], [-1, -1], 'k--')
# plt.plot([0, 0], [-2.2, 3.2], 'k-')
# plt.grid(True)
# plt.title(r"ELU activation function ($\alpha=1$)", fontsize=14)
# plt.axis([-5, 5, -2.2, 3.2])

# save_fig("elu_plot")
# plt.show()


def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * elu(z, alpha)

# tf.nn.selu()

plt.plot(z, selu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1.758, -1.758], 'k--')
plt.plot([0, 0], [-2.2, 3.2], 'k-')
plt.grid(True)
plt.title(r"SELU activation function", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])

save_fig("selu_plot")
plt.show()


###batch_normalization
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

training = tf.placeholder_with_default(False, shape=(), name='training')

hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
bn1_act = tf.nn.elu(bn1)

hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
bn2_act = tf.nn.elu(bn2)

logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
logits = tf.layers.batch_normalization(logits_before_bn, training=training,
                                       momentum=0.9)


#gradient clipping
n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
    logits = tf.layers.dense(hidden5, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
threshold = 1.0
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# training_op = optimizer.minimize(loss) #相当于把这一步拆开
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
              for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(capped_gvs)




#本文件包含几个激活函数和它的图像，batch_norm, 梯度剪裁

