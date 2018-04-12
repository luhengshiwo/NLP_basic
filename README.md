Tensorflow 框架检索
====
# data文件夹保存了一些基本数据
# 1 基本操作 basictensor_handson文件夹
## 1.1 basic_operations.py
* 包含一些基本的tensorflow操作
## 1.2 logistic_regression_handon.py
* 一个简单的tensorflow代码
## 1.3 logistic_fancy.py
* 在基础上加上了额外的features
* 提供了一个随机batch的方法
* 加上了name_scope
* 加上了tensorboard
* 加上了check_point
## 1.4 search_parameters.py
* 提供了一个自动随机搜索参数的方法
# 2 wordembeddings_handson文件夹
## 2.1 jieba(to do !!!!!!)
* train.conll 是文本预处理后
## 2.2 word2vec_handson
* process_data.py 预处理数据，后续移动到结巴目录，并将结巴目录改为数据预处理目录(to do !!!!!!!!!)
* build_word2vec.py
## 2.3 glove_handson
* build_glove.py 明磊贡献的glove生成方法(to do !!!!!!!!!!)
# 3 如何训练一个小型的神经网络small_net_work_handson文件夹
## 3.1 dnn_plain_handson.py
* 一个基础的多层神经网络
## 3.2 dnn_fancy.py
* 在基础上加入name_scope
* tensorbaord
* check_point
* restore
* 保存最优模型的方法
* early_stop
* 中断重启机制
# 4 一个深度神经网络 deep_beural_nets_handson文件夹
## 4.1 basic_operation.py
* 几个激活函数的图
* batch_normlization
* 梯度剪裁的方法
## 4.2 resuing_pretrained_layers.py
* 使用已经训练好的模型
## 4.3 resuing_other_frameworks.py
* 使用其它框架生成模型的参数
## 4.4 learning_rate_decay_handson.py
* 学习率衰减，注意AdaGrad RMSProp Adam 不需要learning_rate_decay
## 4.5 regularization_and_dropout.py
* 如何进行正则化和drop_out
## 4.6 max_norm_regularization_handson.py
* 如何将权重的范数限定在一定范围内
## 4.7 transfer_learning文件夹
* origin_learning_handson.py
  * 一个简单的6层神经网络，为了输出一个保存好的模型
* transfer_learning_handson.py
  * 使用之前训练好的layers，加上新定义的层，去做transfer_learning
* transfer_learning_and_freeze.py
  * 提供了多种方法去transfer_learning,可以选择是否freeze掉已训练的隐层
* catch_frozen_layers.py
  * 提供一个方法，可以将之前隐层的结果缓存下来，提高速度
## 4.8 exercise文件夹
### deep_learning文件夹
* exercise8.1
### transfer_learning_exercise文件夹
* 重用8.1
* freeze掉前面几层
* 缓存freeze的层，速度确实快了
### pretraining_on_auxiliary_task文件夹
#### basemodel.py
* 训练一个模型，有两个dnn，并输入两个图片，比较两个图片是否为一个手写体









