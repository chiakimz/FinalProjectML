from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

from cancer import cancer

cancer = Cancer()
model = cancer.model
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
checkpoint_dir = './cancer_model'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())

root.restore(tf.train.latest_checkpoint(checkpoint_dir))

data_array = []
while True:
    data = (input("Please enter features to predict (type 'exit' to stop):\n"))
    if data == 'exit':
        break
    data = data.split(",")
    data_array.append([float(i) for i in data])
if len(data_array) > 0:
    cancer.predict(data_array)
