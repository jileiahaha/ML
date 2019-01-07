import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b

# 通过log_device_placement参数来记录运行每一个运算的设备。
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))


# 通过tf.device将运算指定到特定的设备上。
with tf.device('/cpu:0'):
	a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
	b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')

with tf.device('/gpu:1'):
	c = a + b


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))





