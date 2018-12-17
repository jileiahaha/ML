# 三层简单神经网络
import tensorflow as tf

# 定义变量
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.constant([[0.7, 0.9]])

# 定义前向传播的神经网络
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 调用会话输出结果
with tf.Session() as sess:
    # 初始w1和w2的方法1
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    # 方法2,：初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y))

