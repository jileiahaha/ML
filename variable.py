from __future__ import print_function
import tensorflow as tf

# 在TensorFlow中，定义了x是变量，他才能是变量，与Python不同
# 定义语法stat = tf.Varibale()
state = tf.Variable(0, name='counter')
# print(state.name) #这里变量没有初始化，所以不会有作用
#定义常量one
one = tf.constant(1)
#定义加法步骤
new_value = tf.add(state, one)
#将state更新成new_value
update = tf.assign(state, new_value)


# 定义了变量之后，一定要初始化变量
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))