# -*- coding: utf-8 -*-
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Direcotry for storing mnist data")

FLAGS = flags.FLAGS
def main(_):
    print(FLAGS.data_dir)

if __name__ == "__main__":
    tf.app.run()
