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
            ip_shape=(2, None), layer_sizes=layer_sizes, activations=["relu", "sigmoid"]
        )

        params = {
            "W1": model.layers[0].weights,
            "b1": model.layers[0].biases,
            "W2": model.layers[1].weights,
            "b2": model.layers[1].biases,
        }

        activations = {
            "A1": ReLU(),
            "A2": Sigmoid(),
        }

        self.X = X
        self.Y = Y
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

    def test_str(self):
        string = str(self.model)
        layers = ", ".join([str(layer) for layer in self.model.layers])
        expected = (
            f"{self.model.__class__.__name__}(Input(ip_shape=(2, None)), {layers})"
        )

        self.assertEqual(string, expected)

        with self.subTest():
            repr_str = repr(self.model)
            self.assertEqual(repr_str, expected)

    def test_initializers_error(self):
        with self.assertRaises(AttributeError):
            _ = Model(
                ip_shape=(2, None),
                layer_sizes=[5, 1],
                activations=["relu", "sigmoid"],
                initializers=["he"],
            )

    def test_no_initializers_passed(self):
        self.assertTrue(all(x is None for x in self.model.initializers))

    def test_initializers_passed(self):
        model = Model(
            ip_shape=(2, None),
            layer_sizes=[5, 3, 3, 1],
            activations=["relu", "tanh", "relu", "sigmoid"],
            initializers=[None, "xavier", None, "xavier"],
        )

        self.assertEqual(model.layers[0].initializer, "he")
        self.assertEqual(model.layers[1].initializer, "xavier")
        self.assertEqual(model.layers[2].initializer, "he")
        self.assertEqual(model.layers[3].initializer, "xavier")

    def test_mismatch_layer_activations(self):
        with self.assertRaises(AttributeError):
            _ = Model(
                ip_shape=(1, 1),
                layer_sizes=[1, 1],
                activations=["relu"],
            )

    def test_build_model(self):
        model = self.model._build_model()

        self.assertIsInstance(model, tuple)
        self.assertEqual(len(model), self.X.shape[0])

        for idx, layer in enumerate(model):
            with self.subTest(layer=layer):
                self.assertEqual(layer.units, self.layer_sizes[idx])

                activation_cls = self.activations[f"A{idx + 1}"].__class__
                self.assertIsInstance(layer.activation, activation_cls)

    def test_predict(self):
        model_preds = self.model.predict(self.X)
        self.forward_propagation()

        for idx, layer in enumerate(self.model.layers):
            with self.subTest(layer=layer):
                np.testing.assert_allclose(layer.linear, self.forward[f"Z{idx + 1}"])
                np.testing.assert_allclose(
                    layer.activations, self.forward[f"A{idx + 1}"]
                )

        np.testing.assert_allclose(model_preds, self.forward["A2"])
