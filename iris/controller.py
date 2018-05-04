from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

from iris import Iris

iris = Iris()
raefe = iris.train()
iris.graph(raefe, 'graphs')
