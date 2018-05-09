from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

from diabetes import Diabetes

diabetes = Diabetes()
model = diabetes.model
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
checkpoint_dir = './diabetes_model'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())

root.restore(tf.train.latest_checkpoint(checkpoint_dir))

diabetes.train()

model = diabetes.model
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
checkpoint_dir = './diabetes_model'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())
root.save(file_prefix=checkpoint_prefix)
