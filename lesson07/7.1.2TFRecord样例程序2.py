import tensorflow as tf

# 2. 读取TFRecord文件
# 读取TFRecord文件中的样例。
reader = tf.TFRecordReader()
# 创建一个队列，维护输入文件列表
filename_queue = tf.train.string_input_producer(["output.tfrecords"])
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

# tf.decode_raw可以将字符串解析成图像对应的像素组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

# 每次运行可以读取TFRecord文件中的一个样例
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
    print(image, label, pixel)
