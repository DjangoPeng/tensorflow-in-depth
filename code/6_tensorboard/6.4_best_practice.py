# -*- coding:utf-8 -*-
import argparse
import sys
import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main(_):
    # 创建日志目录
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # 读取MNIST数据集
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True,
                                      fake_data=FLAGS.fake_data)

    # 创建嵌入变量，保存测试集中的10000张手写体数字图像
    embedding_var = tf.Variable(tf.stack(mnist.test.images[:10000]),
                                trainable=False, name='embedding')
    # 创建交互式会话，并初始化全局变量
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 创建saver, 并保存嵌入变量
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(FLAGS.log_dir + '/model.ckpt'))

    # 创建元信息文件,并写入测试集中10000张手写体数字的标签
    metadata_file = FLAGS.log_dir + '/metadata.tsv'
    with open(metadata_file, 'w') as f:
      for i in range(FLAGS.max_nums):
        c = np.nonzero(mnist.test.labels[::1])[1:][0][i]
        f.write('{}\n'.format(c))
    
    # 创建FileWriter
    writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # 创建投影配置参数
    config = projector.ProjectorConfig()
    embeddings= config.embeddings.add()
    embeddings.tensor_name = 'embedding:0'
    embeddings.metadata_path = os.path.join(FLAGS.log_dir + '/metadata.tsv')

    # 设置全景图文件路径和手写体数字图像的尺寸
    embeddings.sprite.image_path = os.path.join('/tmp/summary/images/mnist_10k_sprite.png')
    embeddings.sprite.single_image_dim.extend([28, 28])

    # 将参数配置写入新创建的投影参数配置文件中
    # TensorBoard启动时会自动加载该文件
    projector.visualize_embeddings(writer, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_nums', type=int, default=10000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/tmp/summary/embeddings',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
