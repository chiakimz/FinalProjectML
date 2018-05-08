from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

import sys
sys.path.insert(0, './iris_flower_categoriser')
from iris import Iris
sys.path.insert(0, './creditcard_fraud_detector')
from fraud import Fraud

iris = Iris()
trained_iris = iris.train()
iris.graph(trained_iris, './iris_flower_categoriser/iris_graphs')
model = iris.model
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
checkpoint_dir = './iris_flower_categoriser/iris_model'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())
root.save(file_prefix=checkpoint_prefix)
