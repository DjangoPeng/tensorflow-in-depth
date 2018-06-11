# -*- coding:utf-8 -*-
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, [None, 784], name='x-input')
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('input_reshape'):
  # 将输入图像x转换成四阶张量
  image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
  # 添加获取手写体图像的汇总操作，设置最大生成10张图像
  tf.summary.image('input', image_shaped_input, 10)

with tf.name_scope('softmax_layer'):
  with tf.name_scope('weights'):
    weights = tf.Variable(tf.zeros([784, 10]))
    # 添加获取模型权重值的汇总操作
    tf.summary.histogram('weights', weights)
  with tf.name_scope('biases'):
    biases = tf.Variable(tf.zeros([10]))
    # 添加获取模型偏置值的汇总操作
    tf.summary.histogram('biases', biases)
  with tf.name_scope('Wx_plus_b'):
    y = tf.matmul(x, weights) + biases
  
with tf.name_scope('cross_entropy'):
  diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
    # 添加获取交叉熵的汇总操作
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(0.001).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 添加获取准确率的汇总操作
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
sess = tf.InteractiveSession()

train_writer = tf.summary.FileWriter('/tmp/summary/mnist' + '/train', sess.graph)
test_writer = tf.summary.FileWriter('/tmp/summary/mnist' + '/test')
tf.global_variables_initializer().run()

def feed_dict(train):
  """填充训练数据或测试数据的方法"""
  if train:
    xs, ys = mnist.train.next_batch(100, fake_data=False)
  else:
    xs, ys = mnist.test.images, mnist.test.labels
  return {x: xs, y_: ys}

for i in range(1000):
  if i % 10 == 0:  # 写汇总数据和测试集的准确率
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # Record train set summaries, and train
    if i % 100 == 99:  # 写运行时的事件数据
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      summary, _ = sess.run([merged, train_step],
                            feed_dict=feed_dict(True),
                            options=run_options,
                            run_metadata=run_metadata)
      train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      train_writer.add_summary(summary, i)
      print('Adding run metadata for', i)
    else:  # 写汇总数据
      summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
      train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()
