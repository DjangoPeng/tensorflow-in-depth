# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
FLAGS = flags.FLAGS
def main(_):
  # 导入数据
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # ...省略中间步骤...

if __name__ == "__main__":
  tf.app.run()
