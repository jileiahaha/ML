
# Session 是 Tensorflow 为了控制,和输出文件的执行的语句.
# 运行 session.run() 可以获得你要得知的运算结果, 或者是你所要运算的部分.
# 在tensorflow中，只有session.run()才会执行我们定义的运算

from __future__ import print_function
import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)
# 因为 product 不是直接计算的步骤,
# 所以我们会要使用 Session 来激活 product 并得到计算结果.
#  有两种形式使用会话控制 Session 。
# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)