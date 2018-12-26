import tensorflow as tf
import numpy as np
import threading
import time


# 每隔1秒判断是否需要停止并打印自己的ID
def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stoping from id: %d\n" % worker_id)
            coord.request_stop()
        else:
            print("Working on id: %d\n" % worker_id)
        time.sleep(1)


# 创建、启动并退出线程
coord = tf.train.Coordinator()
threads = [threading.Thread(target=MyLoop, args=(coord, i, )) for i in range(5)]
for t in threads:t.start()
coord.join(threads)
