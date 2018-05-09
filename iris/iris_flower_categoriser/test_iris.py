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

    def test_download_test_data_function(self):
        filepath = self.iris.download_test_data()[-13:]
        self.assertAllEqual(filepath ,'iris_test.csv')

    def test_format_data_features_and_label_are_tensors(self):
        features, label, dataset = self.iris.format_data(self.iris.download_data())
        self.assertTrue(isinstance(features[0], tf.Tensor))
        self.assertTrue(isinstance(label[0], tf.Tensor))

    def test_test_function_returns_accuracy(self):
        test_accuracy_result = self.iris.test()
        self.assertAllEqual(test_accuracy_result[-1:], '%')

    def test_train_function_adds_to_loss_and_accuracy_array(self):
        train_loss_results, train_accuracy_results = self.iris.train()
        self.assertAllEqual(len(train_loss_results), 201)
        self.assertAllEqual(len(train_accuracy_results), 201)

    def test_predict_function(self):
        self.iris.train()
        possible_answers = ['Example 1 prediction: Iris setosa', 'Example 1 prediction: Iris versicolor', 'Example 1 prediction: Iris virginica']
        predictions = self.iris.predict([[5.1, 3.3, 1.7, 0.5]])
        self.assertTrue(predictions in possible_answers)

    def test_graph_creates_file(self):
        self.iris.graph([[3,2,4,5], [2,7,1,0]], 'test_graphs')
        my_file = Path('./test_graphs/figure.png')
        self.assertTrue(my_file.is_file())


if __name__ == '__main__':
    tf.test.main()
