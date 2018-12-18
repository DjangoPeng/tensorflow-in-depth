# -*- coding:utf-8 -*-
# 类别标签为1字节
LABEL_BYTES = 1
# 图片尺寸为32字节
IMAGE_SIZE = 32
# 图片为RGB 3通道
IMAGE_DEPTH = 3
# 图片数据为32x32x3＝3072字节
IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEPTH
# 10类标签
NUM_CLASSES = 10

import tensorflow as tf

def read_cifar10(data_file, batch_size):
  """从CIFAR-10数据文件读取批样例
  输入参数:
    data_file: CIFAR-10数据文件
    batch_size: 批数据大小
  返回值:
    images: 形如[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]的图像批数据
    labels: 形如[batch_size，NUM_CLASSES]的标签批数据
  """
  # 单条数据记录大小为1+3072=3073字节
  record_bytes = LABEL_BYTES + IMAGE_BYTES
  # 创建文件名列表
  data_files = tf.gfile.Glob(data_file)
  # 创建文件名队列
  file_queue = tf.train.string_input_producer(data_files, shuffle=True)
  # 创建二进制文件对应的Reader实例，按照记录大小从文件名队列中读取样例
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, value = reader.read(file_queue)
  # 将样例拆分为类别标签和图片
  record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
  label = tf.cast(tf.slice(record, [0], [LABEL_BYTES]), tf.int32)
  # 将长度为[depth * height * width]的字符串转换为形如[depth, height, width]的图片张量
  depth_major = tf.reshape(tf.slice(record, [LABEL_BYTES], [IMAGE_BYTES]),
                           [IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE])
  # 改变图片张量各维度顺序，从[depth, height, width]转换为[height, width, depth]
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
  # 创建样例队列
  example_queue = tf.RandomShuffleQueue(
      capacity=16 * batch_size,
      min_after_dequeue=8 * batch_size,
      dtypes=[tf.float32, tf.int32],
      shapes=[[IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH], [1]])
  num_threads = 16
  # 创建样例队列的入队操作
  example_enqueue_op = example_queue.enqueue([image, label])
  # 将定义的16个线程全部添加到queue runner中
  tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
      example_queue, [example_enqueue_op] * num_threads))

  # 从样例队列中读取批样例图片和标签
  images, labels = example_queue.dequeue_many(batch_size)
  labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
  labels = tf.sparse_to_dense(
      tf.concat(values=[indices, labels], axis=1),
      [batch_size, NUM_CLASSES], 1.0, 0.0)

  # 展示images和labels的数据结构
  assert len(images.get_shape()) == 4
  assert images.get_shape()[0] == batch_size
  assert images.get_shape()[-1] == 3
  assert len(labels.get_shape()) == 2
  assert labels.get_shape()[0] == batch_size
  assert labels.get_shape()[1] == NUM_CLASSES

  return images, labels