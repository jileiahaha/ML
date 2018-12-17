# 张量的概念
import tensorflow as tf

# 初始化变量a和b
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print(result) #此时不会直接给出结果，而是给出计算相关参数

# 这里通过会话，显示结果
sess = tf.InteractiveSession ()
print(result.eval())
sess.close()

