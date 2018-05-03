import unittest
import iris
from iris import Iris
import tensorflow as tf

class IrisTest(tf.test.TestCase):

    def setUp(self):
        self.iris = Iris()

    def test_download_function(self):
        filepath = self.iris.download_data()[-17:]
        self.assertAllEqual(filepath ,'iris_training.csv')

    def test_format_data_features_is_tensor(self):
        features, label, dataset = self.iris.format_data()
        self.assertTrue(isinstance(features[0], tf.Tensor))

    def test_format_data_label_is_tensor(self):
        features, label, dataset = self.iris.format_data()
        self.assertTrue(isinstance(label[0], tf.Tensor))

    def test_train_function_adds_to_loss_array(self):
        train_loss_results = self.iris.train()[0]
        self.assertAllEqual(len(train_loss_results), 201)

    def test_train_function_adds_to_accuracy_array(self):
        train_accuracy_results = self.iris.train()[1]
        self.assertAllEqual(len(train_accuracy_results), 201)



if __name__ == '__main__':
    tf.test.main()
