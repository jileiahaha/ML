import tensorflow as tf

# 用于计算滑动平均，初始为0
v1 = tf.Variable(0, dtype=tf.float32)
# 用于模拟神经网络迭代的次数，动态控制衰减率
step = tf.Variable(0, trainable=False)
# 定义一个滑动平均的类
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作，每次执行这个操作时，列表中的变量会更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新变量v1的取值
    sess.run(tf.assign(v1, 5))
    sess.run(maintain_averages_op)  # 衰减率=min{0.99，(1+step)/(10+step)=0.1}=0.1 滑动平均=0.1*0+0.9*5=4.5
    print(sess.run([v1, ema.average(v1)]))

    # 更新step和v1的取值
    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)  # 衰减率=min{0.99，(1+step)/(10+step)=0.999}=0.9 滑动平均=0.99*4.5+0.01*5=4.555
    print(sess.run([v1, ema.average(v1)]))

    # 更新一次v1的滑动平均值
    sess.run(maintain_averages_op)  # 衰减率=min{0.99，(1+step)/(10+step)=0.999}=0.9 滑动平均=0.99*4.555+0.01*5=4.60945
    print(sess.run([v1, ema.average(v1)]))
