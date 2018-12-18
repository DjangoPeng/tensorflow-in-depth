"""5.2_best_practice.py"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_string("train_dir", "/tmp/mnist-log",
                    "Directory for storing checkpoint and summary files")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 0, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update "
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_string("ps_hosts","localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = flags.FLAGS


IMAGE_PIXELS = 28


def main(unused_argv):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")

  # 解析ps和worker的主机名列表
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")

  # 计算worker的数量
  num_workers = len(worker_spec)

  cluster = tf.train.ClusterSpec({
      "ps": ps_spec,
      "worker": worker_spec})
  
  # 如果是ps，直接启动服务，并开始监听worker发起的请求
  if FLAGS.job_name == "ps":
      server.join()

  # 判断当前是否为chief worker的任务进程
  is_chief = (FLAGS.task_index == 0)

  if FLAGS.num_gpus > 0:
    # 假设每台机器的 GPU 数量都相同时，为每台机器的每个 GPU 依次分配一个计算任务。
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    # 如果没有 GPU，直接将计算任务分配到 CPU
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)

  # 根据TensorFlow集群的定义和当前设备的信息，放置对应的模型参数和计算操作
  with tf.device(
      tf.train.replica_device_setter(
          worker_device=worker_device,
          ps_device="/job:ps/cpu:0",
          cluster=cluster)):
      global_step = tf.Variable(0, name="global_step", trainable=False)

      # 隐层模型参数
      hid_w = tf.Variable(
          tf.truncated_normal(
              [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
              stddev=1.0 / IMAGE_PIXELS),
          name="hid_w")
      hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

      # softmax层模型参数
      sm_w = tf.Variable(
          tf.truncated_normal(
              [FLAGS.hidden_units, 10],
              stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
          name="sm_w")
      sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

      # 根据任务编号放置对应的placeholder
      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
      y_ = tf.placeholder(tf.float32, [None, 10])
      # tf.nn.xw_plus_b即为matmul(x, w) + b
      hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
      # 使用relu作为激活函数，hid为隐层输出
      hid = tf.nn.relu(hid_lin)
      # 定义softmax层的输出y，即推理计算出的标签值
      y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
      # 使用交叉熵评估两个概率分布间的相似性。因为概率取值范围为[0, 1]，
      # 同时避免出现无意义的log(0)，所以裁剪y值到区间[1e-10, 1.0]
      cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
      # 使用Adam做最优化求解
      opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    
    # 如果使用同步训练机制
    if FLAGS.sync_replicas:
      # 如果用户没有输入并行副本数，则令其等于worker任务数
      if FLAGS.replicas_to_aggregate is None:
        replicas_to_aggregate = num_workers
      # 如果用户输入了并行副本数，则赋值为命令行解析的并行副本数
      else:
        replicas_to_aggregate = FLAGS.replicas_to_aggregate
      # 创建同步优化器实例，负责计算梯度和更新模型参数
      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=num_workers,
          name="mnist_sync_replicas")
    # 单步训练操作，即利用同步优化器最优化交叉熵
    train_op = opt.minimize(cross_entropy, global_step=global_step)

    # 使用同步训练机制
    if FLAGS.sync_replicas:
      # 其它worker：为local_step设置初始值
      local_init_op = opt.local_step_init_op
      # chief worker：为global_step设置初始值
      if is_chief:
        local_init_op = opt.chief_init_op
      # 定义为未初始化的Variable设置初始值的操作
      ready_for_local_init_op = opt.ready_for_local_init_op

      # 定义启动同步标记队列的QueueRunner实例
      chief_queue_runner = opt.get_chief_queue_runner()
      # 定义为同步标记队列入队初始值的操作
      sync_init_op = opt.get_init_tokens_op()
    # 定义为全局Variable设置初始值的操作
    init_op = tf.global_variables_initializer()

    # 使用同步训练机制，传入本地初始化相关操作
    if FLAGS.sync_replicas:
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=FLAGS.train_dir,
          init_op=init_op,
          local_init_op=local_init_op,
          ready_for_local_init_op=ready_for_local_init_op,
          recovery_wait_secs=1,
          global_step=global_step)
    # 使用异步更新机制，各worker独自训练，与单机模型一致
    else:
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=FLAGS.train_dir,
          init_op=init_op,
          recovery_wait_secs=1,
          global_step=global_step)
    # 配置分布式会话：
    #     在没有可用的GPU时，将操作放置到CPU
    #     不打印设备放置信息
    #     过滤未绑定在ps和worker上的操作
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,  
        log_device_placement=False, 
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

    # 如果是chief worker，则初始化所有worker的分布式会话
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.task_index)
    # 如果是其它worker，则等待chief worker返回的会话
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)

    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)
    # 如果是同步更新模式，并且当前进程为chief worker
    if FLAGS.sync_replicas and is_chief:
      # 初始化同步标记队列
      sess.run(sync_init_op)
      # 通过queue runner启动3个线程，并运行各自的标准服务
      sv.start_queue_runners(sess, [chief_queue_runner])

    # 记录并打印训练开始前的时间
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)
    # 将local_step赋值为0
    local_step = 0
    while True:
      # 填充训练数据
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      train_feed = {x: batch_xs, y_: batch_ys}
      # 执行单步训练操作
      _, step = sess.run([train_op, global_step], feed_dict=train_feed)
      local_step += 1
      # 记录并打印完成当前单步训练所需的时间
      now = time.time()
      print("%f: Worker %d: training step %d done (global step: %d)" %
            (now, FLAGS.task_index, local_step, step))
      # 如果当前超过最大训练步数，退出训练循环
      if step >= FLAGS.train_steps:
        break
    # 记录并打印训练结束的时间
    time_end = time.time()
    print("Training ends @ %f" % time_end)
    # 总训练时间为两者的时间差
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)

    # 填充验证数据
    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    # 在验证数据集上计算模型的交叉熵
    val_xent = sess.run(cross_entropy, feed_dict=val_feed)
    print("After %d training step(s), validation cross entropy = %g" %
          (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
  tf.app.run()