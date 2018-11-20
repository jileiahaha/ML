# 添加层 def add_layer()
# 神经层里常见的参数通常有weights、biases和激励函数。

# 激励函数运行时激活神经网络中某一部分神经元，
# 将激活信息向后传入下一层的神经系统。
# 激励函数的实质是非线性方程。
# Tensorflow 的神经网络里面处理较为复杂的问题时都会需要运用激励函数activation function 。

from __future__ import print_function
import tensorflow as tf

# 定义添加神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size])) #在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1) #在机器学习中，biases的推荐值不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases #神经网络未激活的值
    if activation_function is None: #当activation_function——激励函数为None时，输出就是当前的预测值
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs