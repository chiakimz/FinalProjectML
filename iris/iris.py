from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

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
        self.class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]

    def download_data(self):
        train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
        train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
        return train_dataset_fp

    def download_test_data(self):
        test_url = "http://download.tensorflow.org/data/iris_test.csv"

        test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)
        return test_fp

    def format_data(self, data_filepath):
        train_dataset = tf.data.TextLineDataset(data_filepath)
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

            for x, y in tfe.Iterator(self.format_data(self.download_data())[2]):
                grads = self.__grad(self.model, x, y)
                self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                               global_step=tf.train.get_or_create_global_step())
                epoch_loss_avg(self.__loss(self.model, x, y))
                epoch_accuracy(tf.argmax(self.model(x), axis=1, output_type=tf.int32), y)
            self.train_loss_results.append(epoch_loss_avg.result())
            self.train_accuracy_results.append(epoch_accuracy.result())

            if epoch % 20 == 0:
                self.__print_report(epoch, epoch_loss_avg.result(), epoch_accuracy.result())

        return self.train_loss_results, self.train_accuracy_results

    def test(self):
        test_accuracy = tfe.metrics.Accuracy()

        for (x, y) in tfe.Iterator(self.format_data(self.download_test_data())[2]):
            prediction = tf.argmax(self.model(x), axis=1, output_type=tf.int32)
            test_accuracy(prediction, y)

        print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
        return "{:.3%}".format(test_accuracy.result())

    def predict(self, data):
        predict_dataset = tf.convert_to_tensor(data)

        predictions = self.model(predict_dataset)
        returned_predictions = []

        for i, logits in enumerate(predictions):
          class_idx = tf.argmax(logits).numpy()
          name = self.class_ids[class_idx]
          returned_predictions.append("Example {} prediction: {}".format(i, name))
        print("\n".join(returned_predictions))
        return "\n".join(returned_predictions)

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

    def __print_report(self, epoch, epoch_loss, epoch_accuracy):
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss, epoch_accuracy))

    def graph(self, train_arg, image_folder_path):
        fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle('Training Metrics')
        self.__loss_plotter(train_arg[0], axes)
        self.__accuracy_plotter(train_arg[1], axes)
        plt.savefig(f'{image_folder_path}/figure.png')

    def __loss_plotter(self, loss_results, axes):
        axes[0].set_ylabel("Loss", fontsize=14)
        axes[0].plot(loss_results)

    def __accuracy_plotter(self, accuracy_results, axes):
        axes[1].set_ylabel("Accuracy", fontsize=14)
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].plot(accuracy_results)
