from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

from iris import Iris

iris = Iris()
iris.train()

w1 = tf.contrib.eager.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.contrib.eager.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver([w1,w2])
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
saver.save(sess, './iris/my_test_model')
