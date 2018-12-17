import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

# 1. pb文件的保存方法
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 导出当前计算图的GraphDef部分，只需要这部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    with tf.gfile.GFile("Saved_model/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())

# 2. 加载pb文件。
with tf.Session() as sess:
    model_filename = "Saved_model/combined_model.pb"

    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))

