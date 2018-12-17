# 在图上定义和使用变量
import tensorflow as tf

# g1为图，定义变量“v”,初始值为0
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", [1], initializer=tf.zeros_initializer())  # 设置初始值为0

# g2为图，定义变量“v”,初始值为1
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", [1], initializer=tf.ones_initializer())  # 设置初始值为1

# 在g1中，读取变量“v”的取值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

# 在g2中，读取变量“v”的取值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))