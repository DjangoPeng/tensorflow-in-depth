"""5.1_best_practice.py""" 
# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_float("learning_rate", 0.5, "Learning rate")
FLAGS = flags.FLAGS

def main(_):
  # 创建MNIST数据集实例
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # 创建模型
  x = tf.placeholder(tf.float32, [None, 784])	# 图像数据
  W = tf.Variable(tf.zeros([784, 10]))			  # 模型权重
  b = tf.Variable(tf.zeros([10]))				      # 模型偏置
  y = tf.matmul(x, W) + b						          # 推理操作
  y_ = tf.placeholder(tf.float32, [None, 10]) # 图像标签
  # 使用交叉熵作为损失值
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  # 创建梯度下降优化器
  optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
  # 定义单步训练操作
  train_op = optimizer.minimize(cross_entropy)
  # 创建Saver
  saver = tf.train.Saver()
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # 最大训练步数
  for i in range(1000): 
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
     # 每100步保存一次模型参数
    if i % 100 == 0:
      saver.save(sess, 'mnist.ckpt')
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print('acc=%s' % sess.run(accuracy, 
                            feed_dict={x: mnist.test.images,
                                       y_: mnist.test.labels}))

if __name__ == '__main__':
  tf.app.run()