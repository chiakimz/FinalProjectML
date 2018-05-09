from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

from iris import Iris
global_step = tf.train.get_or_create_global_step()
summary_writer = tf.contrib.summary.create_file_writer(
    'iris_model', flush_millis=10000)
with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
  iris = Iris()
  trained_iris = iris.train()
  iris.graph(trained_iris, 'test_graphs')
  model = iris.model
  optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
  checkpoint_dir = './iris_model'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  root = tfe.Checkpoint(optimizer=optimizer,
                        model=model,
                        optimizer_step=tf.train.get_or_create_global_step())
  root.save(file_prefix=checkpoint_prefix)

  tf.contrib.summary.scalar("loss", my_loss)
