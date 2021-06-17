import unittest

import numpy as np
from dnn.activations import ReLU
from dnn.utils import activation_factory
from dnn.input_layer import Input
from dnn.layer import Layer


class LayerInvalidInputTestCase(unittest.TestCase):
    def test_invalid_ip_error(self):
        ip = np.array([1, 2, 3, 4])
        with self.assertRaises(AttributeError):
            _ = Layer(ip=ip, units=1, activation="relu")


class LayerInitializerTestCase(unittest.TestCase):
    def setUp(self):
        self.ip = Input(shape=(3, None))

    def test_get_initializer(self):
        denom = 3

        with self.subTest(init="he"):
            layer = Layer(ip=self.ip, units=1, activation="relu")
            self.assertAlmostEqual(layer.get_initializer(denom), 2 / denom)

        with self.subTest(init="xavier"):
            layer = Layer(ip=self.ip, units=1, activation="relu", initializer="xavier")
            self.assertAlmostEqual(layer.get_initializer(denom), 1 / denom)

        with self.subTest(init="xavier_uniform"):
            layer = Layer(
                ip=self.ip, units=1, activation="relu", initializer="xavier_uniform"
            )
            self.assertAlmostEqual(layer.get_initializer(denom), 6 / (denom + 1))


class LayerStrTestCase(unittest.TestCase):
    def test_str(self):
        ip, units, activation = Input(shape=(5, None)), 10, "relu"
        layer = Layer(ip=ip, units=units, activation=activation)
        string = str(layer)
        expected = f"Layer(units={units}, activation=ReLU)"

        self.assertEqual(string, expected)

        with self.subTest():
            repr_str = repr(layer)
            self.assertEqual(repr_str, expected)


class LayerWithInputTestCase(unittest.TestCase):
    def setUp(self):
        self.ip_layer, self.units, activation = Input(shape=(5, None)), 10, "relu"
        self.activation = activation_factory(activation)
        self.layer = Layer(ip=self.ip_layer, units=self.units, activation=activation)

    def test_get_ip_error(self):
        with self.assertRaises(ValueError):
            self.layer.get_ip()

    def test_get_y_dim(self):
        self.assertEqual(self.layer.get_y_dim(), 5)

    def test_get_ip(self):
        self.ip_layer.ip = np.random.randn(5, 4)
        np.testing.assert_allclose(self.layer.get_ip(), self.ip_layer.ip)

    def test_units(self):
        self.assertEqual(self.layer.units, self.units)

    def test_init_activation(self):
        self.assertIsInstance(self.layer.activation, ReLU)

    def test_init_params(self):
        self.assertEqual(
            self.layer.weights.shape, (self.units, self.ip_layer.ip_shape[0])
        )
        self.assertEqual(self.layer.biases.shape, (self.units, 1))

    def test_forward_step(self):
        X = np.random.randn(5, 4)
        linear = np.matmul(self.layer.weights, X) + self.layer.biases
        activations = self.activation.calculate_activations(linear)

        self.ip_layer.ip = X
        _ = self.layer.forward_step()

        self.assertEqual(self.layer.linear.shape, (self.units, X.shape[-1]))
        self.assertEqual(self.layer.linear.shape, self.layer.activations.shape)
        np.testing.assert_allclose(self.layer.linear, linear)
        np.testing.assert_allclose(self.layer.activations, activations)


class LayerWithLayerTestCase(unittest.TestCase):
    def setUp(self):
        self.ip_size = 4
        prev_layer_ip_layer = Input(shape=(3, None))
        prev_layer_ip_layer.ip = np.random.randn(3, self.ip_size)
        prev_layer_units = 5
        activation = "relu"

        self.ip_layer = Layer(prev_layer_ip_layer, prev_layer_units, activation="relu")

        self.units = 10
        self.activation = activation_factory(activation)
        self.layer = Layer(ip=self.ip_layer, units=self.units, activation=activation)

    def test_get_y_dim(self):
        self.assertEqual(self.layer.get_y_dim(), self.ip_layer.units)

    def test_units(self):
        self.assertEqual(self.layer.units, self.units)

    def test_init_activation(self):
        self.assertIsInstance(self.layer.activation, ReLU)

    def test_init_params(self):
        self.assertEqual(self.layer.weights.shape, (self.units, self.ip_layer.units))
        self.assertEqual(self.layer.biases.shape, (self.units, 1))

    def test_get_ip_no_fs(self):
        with self.assertRaises(ValueError):
            self.layer.get_ip()

    def test_get_ip_fs(self):
        _ = self.ip_layer.forward_step()

        ip = self.layer.get_ip()
        self.assertEqual(ip.shape, self.ip_layer.activations.shape)
        np.testing.assert_allclose(ip, self.ip_layer.activations)

    def test_forward_step(self):
        _ = self.ip_layer.forward_step()
        linear = (
            np.matmul(self.layer.weights, self.ip_layer.activations) + self.layer.biases
        )
        activations = self.activation.calculate_activations(linear)

        _ = self.layer.forward_step()

        self.assertEqual(self.layer.linear.shape, (self.units, self.ip_size))
        self.assertEqual(self.layer.linear.shape, self.layer.activations.shape)
        np.testing.assert_allclose(self.layer.linear, linear)
        np.testing.assert_allclose(self.layer.activations, activations)
