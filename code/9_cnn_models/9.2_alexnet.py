# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.
This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton
and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014
Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.
Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)
@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def alexnet_v2_arg_scope(weight_decay=0.0005):
  # 在该参数作用域下的 conv2d 和 fully_connected 的计算中,激活函数默认采用 
  # tf.nn.relu,biases 参数默认采用 0.1 恒定值作为初始化值,
  # weights 的正则化默认采用 L2 范式(可以防止过拟合)。用户可显式指定其他方式
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    # 在该参数作用域下的 conv2d 计算中,补零默认采用 SAME 方式,用户可显式指定其他方式
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      # 在该参数作用域下的 max_pool2d 中,补零默认采用 VALID 方式,用户可显式指定其他方
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2',
               global_pool=False):
  """AlexNet version 2.
  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224 or set
        global_pool=True. To use in fully convolutional mode, set
        spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: the number of predicted classes. If 0 or None, the logits layer
    is omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      logits. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original AlexNet.)
  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0
      or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # tensorflow.contrib.slim.arg_scope(list_ops_or_scope, **kwargs)方法 
    #(tensorflow.contrib.framework.python.ops.argscope(list_ops_or_scope,**kwargs)
    # 方法)的作用是:
    # 当 list_ops_or_scope 为 ops 列表时,
    # 将 kwargs 中指定的键值对作为 ops 列表中每个 Op 的输入参数。
    # 此处,调用 slim.arg_scope 方法,使得卷积(conv2d)、全连接(fully_connected)
    # 和最大池化(max_pool2d)三种算子的输出神经元组成的张量都保存在 end_points_collection 中
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
      net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      net = slim.conv2d(net, 192, [5, 5], scope='conv2')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = slim.conv2d(net, 384, [3, 3], scope='conv3')
      net = slim.conv2d(net, 384, [3, 3], scope='conv4')
      net = slim.conv2d(net, 256, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
      # 利用 conv2d 实现全连接层
      with slim.arg_scope([slim.conv2d],
                          weights_initializer=trunc_normal(0.005),
                          biases_initializer=tf.constant_initializer(0.1)):
        net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                          scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        # 因为前一层已经是全连接层,即特征图的尺寸为 1×1,所以此处卷积核也设置为 1×1
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # 将 end_points_collection 转换为 end_points 字典。该字典中的 key 表示该模型中每一层的名字, 
        # value 表示每一层的输出张量。
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        if global_pool:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        if num_classes:
          net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                             scope='dropout7')
          net = slim.conv2d(net, num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            biases_initializer=tf.zeros_initializer(),
                            scope='fc8')
          if spatial_squeeze:
            # 如果 spatial_squeeze 为 True,则对最后一个全连接层的输出张量的形状进行转换,
            # 这样做的好处是能够去除一些不必要的维度,方便计算。该张量原来的形状为 [N,H,W,C],
            # N 表示当前训练的批大小(batch_size),H 和 W 表示特征图的高和宽,C 表示特征图的通道数 
            # (即分类问题中的类别数)。因为该张量是全连接层的输出,所以 H 和 W 都为 1。
            # 通过 tf.squeeze 方法将 H 和 W 所在的维度去掉,该张量的形状变为 [N,C]
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
          end_points[sc.name + '/fc8'] = net
      return net, end_points
alexnet_v2.default_image_size = 224