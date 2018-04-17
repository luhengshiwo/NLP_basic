import tensorflow as tf
import numpy as np
# sess = tf.Session()
# x = tf.Variable(3,name='x')
# y = tf.Variable(4,name='y')
# z = tf.constant(10,name = 'z')
# x = x**y+z
# init = tf.global_variables_initializer()


x = tf.Variable(3,name='x')
x = x+2
y = x+5
z = x*3
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(y.eval())
    print(z.eval())

with tf.Session() as sess:
    sess.run(init)
    y_eval,z_eval = sess.run([y,z])
    print(y_eval)
    print(z_eval)

#常用的操作有如下
# a = tf.constant([x*x for x in range(1,25)],shape=[2,3,4],dtype=np.float32)
# b = tf.constant([x*x for x in range(1,49)],shape=[2,3,8],dtype=np.float32)
# c = tf.expand_dims(a,1) #扩大维度
# c = tf.reshape(a,shape=[2,1,6]) #重新改变维度
# c = tf.nn.softmax(a,axis=1) #在axis这维上做softmax
# c = tf.squeeze(a,[-1])#去掉指定的dim上的dim为1的那一维，和expand_dims()相反
# c = tf.transpose(a,[1,0,2])#转置
# c = tf.unstack(a,axis=1)#把一个tensor按照axis指定的维度拆开，变成一个list
# c = tf.add_n([a,a])#把两个tensor element-wise的相加
# c = tf.concat([a,b],axis=-1)#在axis指定维度上进行拼接
# d = np.c_[np.array([1,2,3]),np.array([4,5,6])]#变成[[1,4],[2,5],[3,6]]
# d = np.ones((1000, 1))
# print(d)
# with tf.Session() as sess:
#     init.run()
#     d = sess.run(c)
#     # d = c.eval()
#     print(a.eval())
#     print(d)
