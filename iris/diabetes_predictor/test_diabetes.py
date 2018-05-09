from os import path
from pathlib import Path
import unittest
from diabetes import Diabetes
import tensorflow as tf
tf.enable_eager_execution()

class DiabetesTest(tf.test.TestCase):

    def setUp(self):
        self.diabetes = Diabetes()

    def test_download_function(self):
        filepath = self.diabetes.download_data()[-12:]
        self.assertAllEqual(filepath ,'diabetes.csv')

    def test_format_data_features_and_label_are_tensors(self):
        features, label, dataset = self.diabetes.format_data(self.diabetes.download_data())
        self.assertTrue(isinstance(features[0], tf.Tensor))
        self.assertTrue(isinstance(label[0], tf.Tensor))

    def test_train_function_adds_to_loss_and_accuracy_array(self):
        train_loss_results, train_accuracy_results = self.diabetes.train()
        self.assertAllEqual(len(train_loss_results), 500)
        self.assertAllEqual(len(train_accuracy_results), 500)

    def test_test_function_returns_accuracy(self):
        test_accuracy_result = self.diabetes.test()
        self.assertAllEqual(test_accuracy_result[-1:], '%')

    def test_graph_creates_file(self):
        self.diabetes.graph([[3,2,4,5], [2,7,1,0]], 'test_diabetes_graphs')
        my_file = Path('./test_diabetes_graphs/figure.png')
        self.assertTrue(my_file.is_file())


if __name__ == '__main__':
    tf.test.main()
