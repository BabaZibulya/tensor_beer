from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

from tensor_model import get_model

FLAGS = None


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

            # Build model...
            global_step = tf.contrib.framework.get_or_create_global_step()
            X, Y, x_train, y_train, loss_op, train_op, accuracy = get_model()

        # The StopAtStepHook handles stopping after running given steps.
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.steps)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="/workdir/tf_res",
                                               hooks=hooks) as mon_sess:
            i = 0
            while i < FLAGS.steps: # not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See <a href="../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                print('sess', i)
                mon_sess.run([loss_op, train_op], feed_dict={X:x_train, Y:y_train})
                i += 1

# 20 = 10100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
      "--ps_hosts",
      type=str,
      default="10.128.0.2:2223", # instance-1
      # default = "192.168.1.13:2222",
      help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
      "--worker_hosts",
      type=str,
      default="10.128.0.3:2223", #,10.128.0.4:2223,10.10.128.0.5:2223", # instance-2, instance-3, instance-4
      # default = "192.168.1.13:2223",
      help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
    )
    # Number of steps
    parser.add_argument(
      "--steps",
      type=int,
      default=10,
      help="Number of steps"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

