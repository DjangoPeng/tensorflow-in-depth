"""trainer.py"""
# -*- coding: utf-8 -*-
from tensorflow import flags
import tensorflow as tf

# 定义TensorFlow集群参数
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization.")
flags.DEFINE_string("ps_hosts", None,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", None,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or PS")
def main(unused_argv):
  # 解析集群参数ps_hosts和worker_hosts
  PS_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")
  # 定义TensorFlow集群
  cluster = tf.train.ClusterSpec({
      "PS": PS_spec,
      "worker": worker_spec})

  server = tf.train.Server(
      cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  # 启动PS，开始监听各worker的请求
  if FLAGS.job_name == "PS":
    server.join()
  # 将任务编号为0的worker设置为chief worker
  is_chief = (FLAGS.task_index == 0)