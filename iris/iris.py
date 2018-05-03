from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

class Iris:

    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(3)
        ])

    def download_data(self):
        train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
        train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
        return train_dataset_fp

    def parse_csv(self, line):
        example_defaults = [[0.], [0.], [0.], [0.], [0]]
        parsed_line = tf.decode_csv(line, example_defaults)
        features = tf.reshape(parsed_line[:-1], shape=(4,))
        label = tf.reshape(parsed_line[-1], shape=())
        return features, label

    def format_data(self):
        train_dataset = tf.data.TextLineDataset(self.download_data())
        train_dataset = train_dataset.skip(1)
        train_dataset = train_dataset.map(self.parse_csv)
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(32)
        features, label = tfe.Iterator(train_dataset).next()
        return features, label
