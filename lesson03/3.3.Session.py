# 会话的使用
import tensorflow as tf

# 初始化变量a和b
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print(result) #此时不会直接给出结果，而是给出计算相关参数

# 1.创建和关闭会话
# 创建一个会话。
sess = tf.Session()
# 使用会话得到之前计算的结果。
print(sess.run(result))
# 关闭会话使得本次运行中使用到的资源可以被释放。
sess.close()
# 如果由于异常而退出的话，不会运行到close（），这样会导致内存泄漏

# 2.使用上下文（with statement）创建会话
# 这个方法更好，解决了上述问题，
with tf.Session() as sess:
    print(sess.run(result))

# 1.设置默认会话
sess = tf.Session()
with sess.as_default():
     print(result.eval())

# 1.设置默认会话（和上述一样的功能）
sess = tf.Session()
# 下面的两个命令有相同的功能。
print(sess.run(result))
print(result.eval(session=sess))

# 2.使用tf.InteractiveSession构建会话
sess = tf.InteractiveSession ()
print(result.eval())
sess.close()

# 3.通过ConfigProto配置会话
config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)