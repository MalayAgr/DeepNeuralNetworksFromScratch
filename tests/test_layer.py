import unittest

import numpy as np
from dnn.activations import ReLU
from dnn.utils import activation_factory
from dnn.input_layer import Input
from dnn.model import Layer


class LayerInvalidInputTestCase(unittest.TestCase):
    def test_invalid_ip_error(self):
        ip = np.array([1, 2, 3, 4])
        with self.assertRaises(AttributeError):
            _ = Layer(ip=ip, units=1, activation="relu")


class LayerInitializerTestCase(unittest.TestCase):
    def setUp(self):
        self.ip = Input(shape=(3, 4))

    def test_get_initializer(self):
        denominator = self.ip.ip_shape[0]

        with self.subTest(init="he"):
            layer = Layer(ip=self.ip, units=1, activation="relu")
            self.assertAlmostEqual(layer.get_initializer(), 2 / denominator)

        with self.subTest(init="xavier"):
            layer = Layer(ip=self.ip, units=1, activation="relu", initializer="xavier")
            self.assertAlmostEqual(layer.get_initializer(), 1 / denominator)

        with self.subTest(init="xavier_uniform"):
            layer = Layer(
                ip=self.ip, units=1, activation="relu", initializer="xavier_uniform"
            )
            self.assertAlmostEqual(layer.get_initializer(), 6 / (denominator + 1))


class LayerStrTestCase(unittest.TestCase):
    def test_str(self):
        ip, units, activation = Input(shape=(5, 4)), 10, "relu"
        layer = Layer(ip=ip, units=units, activation=activation)
        string = str(layer)
        expected = f"Layer(ip_shape={ip.ip_shape}, units={units}, activation=ReLU)"

        self.assertEqual(string, expected)

        with self.subTest():
            repr_str = repr(layer)
            self.assertEqual(repr_str, expected)


class LayerWithInputTestCase(unittest.TestCase):
    def setUp(self):
        self.ip_layer, self.units, activation = Input(shape=(5, 4)), 10, "relu"
        self.activation = activation_factory(activation)
        self.layer = Layer(ip=self.ip_layer, units=self.units, activation=activation)

    def test_get_ip_shape(self):
        self.assertEqual(self.layer.ip_shape, self.ip_layer.ip_shape)

    def test_get_ip_error(self):
        with self.assertRaises(ValueError):
            self.layer.get_ip()

    def test_get_ip(self):
        self.ip_layer.ip = np.random.randn(5, 4)
        np.testing.assert_allclose(self.layer.get_ip(), self.ip_layer.ip)

    def test_units(self):
        self.assertEqual(self.layer.units, self.units)

    def test_train_size(self):
        self.assertEqual(self.layer.train_size, self.ip_layer.ip_shape[-1])

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

        self.assertEqual(
            self.layer.linear.shape, (self.units, self.ip_layer.ip_shape[-1])
        )
        self.assertEqual(self.layer.linear.shape, self.layer.activations.shape)
        np.testing.assert_allclose(self.layer.linear, linear)
        np.testing.assert_allclose(self.layer.activations, activations)


class LayerWithLayerTestCase(unittest.TestCase):
    def setUp(self):
        prev_layer_ip_layer = Input(shape=(3, 4))
        prev_layer_ip_layer.ip = np.random.randn(3, 4)
        prev_layer_units = 5
        activation = "relu"

        self.ip_layer = Layer(prev_layer_ip_layer, prev_layer_units, activation="relu")

        self.units = 10
        self.activation = activation_factory(activation)
        self.layer = Layer(ip=self.ip_layer, units=self.units, activation=activation)

    def test_get_ip_shape(self):
        x_dim, y_dim = self.ip_layer.units, self.ip_layer.train_size
        self.assertEqual(self.layer.ip_shape, (x_dim, y_dim))

    def test_units(self):
        self.assertEqual(self.layer.units, self.units)

    def test_train_size(self):
        self.assertEqual(self.layer.train_size, self.ip_layer.ip_shape[-1])

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

        self.assertEqual(
            self.layer.linear.shape, (self.units, self.ip_layer.ip_shape[-1])
        )
        self.assertEqual(self.layer.linear.shape, self.layer.activations.shape)
        np.testing.assert_allclose(self.layer.linear, linear)
        np.testing.assert_allclose(self.layer.activations, activations)


class OutputLayerGradientTestCase(unittest.TestCase):
    def setUp(self):
        train_size = 5

        op_layer_ip = Input(shape=(4, train_size))
        op_layer_ip.ip = ReLU().calculate_activations(np.random.randn(4, train_size))
        op_layer = Layer(ip=op_layer_ip, units=1, activation="sigmoid")
        _ = op_layer.forward_step()

        Y = np.random.choice((0, 1), size=(1, train_size))

        self.train_size = train_size
        self.op_layer = op_layer
        self.Y = Y
        self.dA_params = self.binary_cross_entropy_derivative()

    def binary_cross_entropy_derivative(self):
        Y = self.Y
        A = self.op_layer.activations
        return ((1 - Y) / (1 - A)) - (Y / A)

    def test_get_layer_dA(self):
        np.testing.assert_allclose(
            self.op_layer.get_layer_dA(self.dA_params), self.dA_params
        )

    def test_gradients(self):
        self.op_layer.layer_gradients(self.dA_params)

        dZ = self.op_layer.activations - self.Y
        dW = np.matmul(dZ, self.op_layer.get_ip().T) / self.train_size
        db = np.sum(dZ, axis=1, keepdims=True) / self.train_size

        op_dW = self.op_layer.gradients["weights"]
        op_db = self.op_layer.gradients["biases"]

        self.assertEqual(op_dW.shape, dW.shape)
        self.assertEqual(op_db.shape, db.shape)

        np.testing.assert_allclose(op_dW, dW)
        np.testing.assert_allclose(op_db, db)

        with self.subTest():
            W = self.op_layer.weights - 0.01 * dW
            b = self.op_layer.biases - 0.01 * db

            self.op_layer.update_params(0.01)

            np.testing.assert_allclose(self.op_layer.weights, W)
            np.testing.assert_allclose(self.op_layer.biases, b)


class HiddenLayerGradientTest(unittest.TestCase):
    def setUp(self):
        train_size = 5

        ip_layer, units, activation = Input(shape=(5, train_size)), 10, "relu"
        ip_layer.ip = np.random.randn(5, train_size)

        self.layer = Layer(ip=ip_layer, units=units, activation=activation)
        self.layer.forward_step()

        op_layer = Layer(ip=self.layer, units=1, activation="sigmoid")
        _ = op_layer.forward_step()

        Y = np.random.choice((0, 1), size=(1, train_size))
        dA_params = ((1 - Y) / (1 - op_layer.activations)) - (Y / op_layer.activations)

        _ = op_layer.layer_gradients(dA_params)

        self.train_size = train_size
        self.op_layer = op_layer

    def test_get_layer_dA(self):
        W = self.op_layer.weights
        dZ = self.op_layer.dZ
        dA = np.matmul(W.T, dZ)

        layer_dA = self.layer.get_layer_dA(self.op_layer)
        self.assertEqual(layer_dA.shape, dA.shape)
        np.testing.assert_allclose(layer_dA, dA)

    def test_gradients(self):
        self.layer.layer_gradients(self.op_layer)

        dZ = np.matmul(
            self.op_layer.weights.T, self.op_layer.dZ
        ) * self.layer.activation.calculate_derivatives(self.layer.linear)

        dW = np.matmul(dZ, self.layer.get_ip().T) / self.train_size
        db = np.sum(dZ, axis=1, keepdims=True) / self.train_size

        layer_dW = self.layer.gradients["weights"]
        layer_db = self.layer.gradients["biases"]

        self.assertEqual(layer_dW.shape, dW.shape)
        self.assertEqual(layer_db.shape, db.shape)

        np.testing.assert_allclose(layer_dW, dW)
        np.testing.assert_allclose(layer_db, db)

        with self.subTest():
            W = self.layer.weights - 0.01 * dW
            b = self.layer.biases - 0.01 * db

            self.layer.update_params(0.01)

            np.testing.assert_allclose(self.layer.weights, W)
            np.testing.assert_allclose(self.layer.biases, b)
