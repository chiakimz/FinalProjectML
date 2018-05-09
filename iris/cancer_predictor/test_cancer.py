from os import path
from pathlib import Path
import unittest
from cancer import Cancer
import tensorflow as tf
tf.enable_eager_execution()

class CancerTest(tf.test.TestCase):

    def setUp(self):
        self.cancer = Cancer()

    def test_download_function(self):
        filepath = self.cancer.download_data()
        self.assertAllEqual(filepath ,'cancer-training.csv')

    def test_download_test_data_function(self):
        filepath = self.cancer.download_test_data()
        self.assertAllEqual(filepath ,'cancer-testing.csv')

    def test_format_data_features_is_tensor(self):
        features, label, dataset = self.cancer.format_data(self.cancer.download_data())
        self.assertTrue(isinstance(features[0], tf.Tensor))
        self.assertTrue(isinstance(label[0], tf.Tensor))

    def test_test_function_returns_accuracy(self):
        test_accuracy_result = self.cancer.test()
        self.assertAllEqual(test_accuracy_result[-1:], '%')

    def test_train_function_adds_to_loss_array(self):
        train_loss_results, train_accuracy_results = self.cancer.train()
        self.assertAllEqual(len(train_loss_results), 400)
        self.assertAllEqual(len(train_accuracy_results), 400)

    def test_predict_function(self):
        self.cancer.train()
        possible_answers = ['Example 1 prediction: Benign', 'Example 1 prediction: Malignant']
        predictions = self.cancer.predict([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]])
        self.assertTrue(predictions in possible_answers)

    def test_graph_creates_file(self):
        self.cancer.graph([[3,2,4,5], [2,7,1,0]], 'test_cancer_graphs')
        my_file = Path('./test_cancer_graphs/figure.png')
        self.assertTrue(my_file.is_file())

if __name__ == '__main__':
    tf.test.main()
