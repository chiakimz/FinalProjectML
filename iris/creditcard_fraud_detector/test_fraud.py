import shutil, tempfile
from os import path
from pathlib import Path
import unittest
from fraud import Fraud
import tensorflow as tf
tf.enable_eager_execution()

class IrisTest(tf.test.TestCase):

    def setUp(self):
        self.fraud = Fraud()

    def test_download_function(self):
        filepath = self.fraud.download_data()[-14:]
        self.assertAllEqual(filepath ,'creditcard.csv')

    def test_format_data_features_is_tensor(self):
        features, label, dataset = self.fraud.format_data(self.fraud.download_data())
        self.assertTrue(isinstance(features[0], tf.Tensor))
        self.assertTrue(isinstance(label[0], tf.Tensor))

    def test_train_function_adds_to_loss_array(self):
        train_loss_results, train_accuracy_result = self.fraud.train()
        self.assertAllEqual(len(train_loss_results), 201)
        self.assertAllEqual(len(train_accuracy_results), 201)

    def test_graph_creates_file(self):
        self.fraud.graph([[3,2,4,5], [2,7,1,0]], 'test_fraud_graphs')
        my_file = Path('./test_graphs/figure.png')
        self.assertTrue(my_file.is_file())

if __name__ == '__main__':
    tf.test.main()
