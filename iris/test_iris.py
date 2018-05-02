import unittest
import iris
from iris import Iris
import os.path
#
# def download_data():
#     pass

class IrisTest(unittest.TestCase):

    def test_download_function(self):
        self.assertEqual(Iris.download_data()[-17:] ,'iris_training.csv')

if __name__ == '__main__':
    unittest.main()
