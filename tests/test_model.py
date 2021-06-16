import unittest

import numpy as np
from dnn.model import Model
from dnn.utils import loss_factory
from dnn.activations import ReLU, Sigmoid


class ModelTestCase(unittest.TestCase):
    def setUp(self):
        X = np.random.randn(2, 10)
        Y = np.random.choice((0, 1), size=(1, 10))

        layer_sizes = (5, 1)

        model = Model(
            ip_shape=X.shape, layer_sizes=layer_sizes, activations=["relu", "sigmoid"]
        )

        params = {
            "W1": model.model[0].weights,
            "b1": model.model[0].biases,
            "W2": model.model[1].weights,
            "b2": model.model[1].biases,
        }

        activations = {
            "A1": ReLU(),
            "A2": Sigmoid(),
        }

        self.X = X
        self.Y = Y
        self.train_size = Y.shape[-1]
        self.lr = 0.1
        self.loss = loss_factory("bse", Y=Y)
        self.params = params
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.model = model

    def forward_propagation(self):
        Z1 = np.matmul(self.params["W1"], self.X) + self.params["b1"]
        A1 = self.activations["A1"].calculate_activations(Z1)

        Z2 = np.matmul(self.params["W2"], A1) + self.params["b2"]
        A2 = self.activations["A2"].calculate_activations(Z2)

        self.forward = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    def backpropagation(self):
        dZ2 = self.forward["A2"] - self.Y
        dW2 = np.matmul(dZ2, self.forward["A1"].T) / self.train_size
        db2 = np.sum(dZ2, axis=1, keepdims=True) / self.train_size

        derivatives = self.activations["A1"].calculate_derivatives(self.forward["Z1"])
        dZ1 = np.matmul(self.params["W2"].T, dZ2) * derivatives
        dW1 = np.matmul(dZ1, self.X.T) / self.train_size
        db1 = np.sum(dZ1, axis=1, keepdims=True) / self.train_size

        self.gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_params(self):
        keys = (("W1", "dW1"), ("W2", "dW2"), ("b1", "db1"), ("b2", "db2"))

        for param, grad in keys:
            self.params[param] -= self.lr * self.gradients[grad]

    def train(self, iterations=10):
        history = []

        for _ in range(iterations):
            self.forward_propagation()
            history.append(self.loss.compute_loss(self.forward["A2"]))
            self.backpropagation()
            self.update_params()

        return history

    def test_str(self):
        string = str(self.model)
        layers = ", ".join([str(layer) for layer in self.model.model])
        expected = (
            f"{self.model.__class__.__name__}(InputLayer{self.X.shape}, {layers})"
        )

        self.assertEqual(string, expected)

        with self.subTest():
            repr_str = repr(self.model)
            self.assertEqual(repr_str, expected)

    def test_mismatch_layer_activations(self):
        with self.assertRaises(AttributeError):
            _ = Model(
                ip_shape=(1, 1),
                layer_sizes=[1, 1],
                activations=["relu"],
            )

    def test_validate_labels_shape(self):
        with self.assertRaises(ValueError):
            Y = np.random.randn(1, 2)
            self.model._validate_labels_shape(self.X, Y)

        with self.assertRaises(ValueError):
            Y = np.random.randn(2, 10)
            self.model._validate_labels_shape(self.X, Y)

    def test_build_model(self):
        model = self.model._build_model()

        self.assertIsInstance(model, tuple)
        self.assertEqual(len(model), self.X.shape[0])

        for idx, layer in enumerate(model):
            with self.subTest(layer=layer):
                self.assertEqual(layer.units, self.layer_sizes[idx])

                activation_cls = self.activations[f"A{idx + 1}"].__class__
                self.assertIsInstance(layer.activation, activation_cls)

    def test_forward_propagation_error(self):
        X = np.random.randn(3, 4)

        with self.assertRaises(ValueError):
            self.model._forward_propagation(X)

    def test_forward_propagation(self):
        model_preds = self.model._forward_propagation(self.X)

        self.forward_propagation()

        for idx, layer in enumerate(self.model.model):
            with self.subTest(layer=layer):
                np.testing.assert_allclose(layer.linear, self.forward[f"Z{idx + 1}"])
                np.testing.assert_allclose(
                    layer.activations, self.forward[f"A{idx + 1}"]
                )

        np.testing.assert_allclose(model_preds, self.forward["A2"])

    def test_backprop(self):
        preds = self.model._forward_propagation(self.X)
        self.model._backpropagation(self.loss, preds)

        self.forward_propagation()
        self.backpropagation()

        for idx, layer in enumerate(self.model.model):
            with self.subTest(layer=layer):
                dW = layer.gradients["weights"]
                db = layer.gradients["biases"]

                np.testing.assert_allclose(dW, self.gradients[f"dW{idx + 1}"])
                np.testing.assert_allclose(db, self.gradients[f"db{idx + 1}"])

    def test_update_params(self):
        preds = self.model._forward_propagation(self.X)
        self.model._backpropagation(self.loss, preds)
        self.model._update_params(self.lr)

        self.forward_propagation()
        self.backpropagation()
        self.update_params()

        for idx, layer in enumerate(self.model.model):
            with self.subTest(layer=layer):
                np.testing.assert_allclose(layer.weights, self.params[f"W{idx + 1}"])
                np.testing.assert_allclose(layer.biases, self.params[f"b{idx + 1}"])

    def test_training(self):
        self.model.train(self.X, self.Y, iterations=10, lr=self.lr, show_loss=False)
        self.train(iterations=10)

        for idx, layer in enumerate(self.model.model):
            with self.subTest(layer=layer):
                np.testing.assert_allclose(layer.weights, self.params[f"W{idx + 1}"])
                np.testing.assert_allclose(layer.biases, self.params[f"b{idx + 1}"])

    def test_predict(self):
        self.model.train(self.X, self.Y, iterations=10, lr=self.lr, show_loss=False)
        self.train(iterations=10)

        preds = self.model.predict(self.X)
        self.forward_propagation()

        np.testing.assert_allclose(preds, self.forward["A2"])


class ModelLossMonotonicityTestCase(unittest.TestCase):
    def monotonically_decreasing(self, x):
        return all(x >= y for x, y in zip(x, x[1:]))

    def test_monotonicity(self):
        X = np.random.randn(5, 100)
        Y = np.random.choice((0, 1), size=(1, 100))

        model = Model(
            ip_shape=X.shape, layer_sizes=(10, 1), activations=["relu", "sigmoid"]
        )

        lrs = [0.001, 0.003, 0.01, 0.03, 0.1]

        for lr in lrs:
            with self.subTest(lr=lr):
                history = model.train(X, Y, iterations=1000, lr=lr, show_loss=False)
                self.assertTrue(self.monotonically_decreasing(history))
