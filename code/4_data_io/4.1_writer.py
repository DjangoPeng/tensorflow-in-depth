# -*- coding: utf-8 -*-
import tensorflow as tf
# 创建向TFRecords文件写数据记录的writer
writer = tf.python_io.TFRecordWriter('stat.tfrecord')
# 2轮循环构造输入样例
for i in range(1,3):
  # 创建example.proto中定义的样例
  example = tf.train.Example(
      features = tf.train.Features(
          feature = {
            'id': tf.train.Feature(int64_list =
                tf.train.Int64List(value=[i])),
            'age': tf.train.Feature(int64_list =
                tf.train.Int64List(value=[i*24])),
            'income': tf.train.Feature(float_list =
                tf.train.FloatList(value=[i*2048.0])),
            'outgo': tf.train.Feature(float_list =
                tf.train.FloatList(value=[i*1024.0]))
          }
      )
  )
  # 将样例序列化为字符串后，写入stat.tfrecord文件
  writer.write(example.SerializeToString())
# 关闭输出流
writer.close()