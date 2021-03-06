from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import datetime

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

from tensor_model import get_model

FLAGS = None

def now_time():
    return datetime.datetime.now() #.strftime("%H:%M:%S")


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")[ :FLAGS.instances]

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        print('START ps time', now_time())
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

            # Build model...
            global_step = tf.train.get_or_create_global_step()
            X, Y, x_train, y_train, loss_op, train_op, accuracy = get_model(tf.train.get_global_step())

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
            ts = now_time()
            tt = ts
            print('START worker time', ts)
            while not mon_sess.should_stop(): # i < FLAGS.steps: # 
                # Run a training step asynchronously.
                # See <a href="../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                mon_sess.run([loss_op, train_op], feed_dict={X:x_train, Y:y_train})
                tn = now_time()
                print('worker after step', i, tn - tt, tn - ts, tn)
                tt = tn
                i += 1
            print('END worker time', tt - ts, tt)

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
      default="10.128.0.3:2223,10.128.0.4:2223,10.10.128.0.5:2223", # instance-2, instance-3, instance-4
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
    # Number of instances
    parser.add_argument(
      "--instances",
      type=int,
      default=1,
      help="Number of instances"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

