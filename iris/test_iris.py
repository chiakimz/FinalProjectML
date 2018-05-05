import shutil, tempfile
from os import path
from pathlib import Path
import unittest
from iris import Iris
import tensorflow as tf
tf.enable_eager_execution()

class IrisTest(tf.test.TestCase):

    def setUp(self):
        self.iris = Iris()

    def test_download_function(self):
        filepath = self.iris.download_data()[-17:]
        self.assertAllEqual(filepath ,'iris_training.csv')

    def test_download_function(self):
        filepath = self.iris.download_test_data()[-13:]
        self.assertAllEqual(filepath ,'iris_test.csv')

    def test_format_data_features_is_tensor(self):
        features, label, dataset = self.iris.format_data(self.iris.download_data())
        self.assertTrue(isinstance(features[0], tf.Tensor))

    def test_format_data_label_is_tensor(self):
        features, label, dataset = self.iris.format_data(self.iris.download_data())
        self.assertTrue(isinstance(label[0], tf.Tensor))

    def test_train_function_returns_accuracy(self):
        test_accuracy_result = self.iris.test()
        self.assertAllEqual(test_accuracy_result[-1:], '%')

    def test_train_function_adds_to_loss_array(self):
        train_loss_results = self.iris.train()[0]
        self.assertAllEqual(len(train_loss_results), 201)

    def test_train_function_adds_to_accuracy_array(self):
        train_accuracy_results = self.iris.train()[1]
        self.assertAllEqual(len(train_accuracy_results), 201)

    def test_graph_creates_file(self):
        self.iris.graph([[3,2,4,5], [2,7,1,0]], 'test_graphs')
        my_file = Path('./test_graphs/figure.png')
        self.assertTrue(my_file.is_file())


if __name__ == '__main__':
    tf.test.main()
