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
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.train_loss_results = []
        self.train_accuracy_results = []
        self.epoch_number = 201

    def download_data(self):
        train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
        train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
        return train_dataset_fp

    def format_data(self):
        train_dataset = tf.data.TextLineDataset(self.download_data())
        train_dataset = train_dataset.skip(1)
        train_dataset = train_dataset.map(self.__parse_csv)
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(32)
        features, label = tfe.Iterator(train_dataset).next()
        return features, label, train_dataset

    def train(self):
        for epoch in range(self.epoch_number):
            epoch_loss_avg = tfe.metrics.Mean()
            epoch_accuracy = tfe.metrics.Accuracy()

            for x, y in tfe.Iterator(self.format_data()[2]):
                grads = self.__grad(self.model, x, y)
                self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                               global_step=tf.train.get_or_create_global_step())
                epoch_loss_avg(self.__loss(self.model, x, y))
                epoch_accuracy(tf.argmax(self.model(x), axis=1, output_type=tf.int32), y)
            self.train_loss_results.append(epoch_loss_avg.result())
            self.train_accuracy_results.append(epoch_accuracy.result())

            if epoch % 20 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

        return self.train_loss_results, self.train_accuracy_results

    def __parse_csv(self, line):
        example_defaults = [[0.], [0.], [0.], [0.], [0]]
        parsed_line = tf.decode_csv(line, example_defaults)
        features = tf.reshape(parsed_line[:-1], shape=(4,))
        label = tf.reshape(parsed_line[-1], shape=())
        return features, label

    def __loss(self, model, x, y):
        y_ = model(x)
        return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

    def __grad(self, model, inputs, targets):
        with tfe.GradientTape() as tape:
            loss_value = self.__loss(model, inputs, targets)
        return tape.gradient(loss_value, model.variables)
