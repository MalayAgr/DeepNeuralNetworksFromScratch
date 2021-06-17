import unittest

import numpy as np
from dnn.input_layer import Input


class InputTestCase(unittest.TestCase):
    def setUp(self):
        self.input_layer = Input(shape=(3, None))

    def test_str(self):
        expected = "Input(ip_shape=(3, None))"
        self.assertEqual(expected, str(self.input_layer))

        with self.subTest():
            self.assertEqual(expected, repr(self.input_layer))

    def test_shape_mismatch(self):
        X = np.random.randn(5, 3)
        with self.assertRaises(AttributeError):
            self.input_layer.ip = X

    def test_setter(self):
        X = np.random.randn(3, 4)
        self.input_layer.ip = X

        np.testing.assert_allclose(X, self.input_layer._ip)
        np.testing.assert_allclose(X, self.input_layer.ip)
