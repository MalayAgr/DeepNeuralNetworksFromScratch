from abc import abstractmethod

from numpy.core.fromnumeric import size
from dnn.loss import Loss, BinaryCrossEntropy, MeanSquaredError
from dnn.activations import ReLU, Sigmoid

import numpy as np

import unittest


class LossTestCase:
    loss_cls = None

    @abstractmethod
    def loss_func(self):
        pass

    @abstractmethod
    def loss_derivatives(self):
        pass

    def setUp(self):
        Y = np.random.choice((0, 1), size=(1, 5))
        preds = Sigmoid().calculate_activations(np.random.randn(1, 5))

        self.loss = self.loss_cls()

        self.Y = Y
        self.preds = preds

    def test_validate_input(self):
        preds = np.random.randn(1, 6)

        with self.assertRaises(AttributeError):
            self.loss.validate_input(self.Y, preds)

    def test_compute_loss(self):
        expected_loss = self.loss_func()
        computed_loss = self.loss.compute_loss(self.Y, self.preds)
        self.assertAlmostEqual(computed_loss, expected_loss)

    def test_compute_derivatives(self):
        expected_derivatives = self.loss_derivatives()

        derivatives = self.loss.compute_derivatives(self.Y, self.preds)

        self.assertEqual(derivatives.shape, expected_derivatives.shape)
        np.testing.assert_allclose(derivatives, expected_derivatives)

    def test_registry_membership(self):
        names = self.loss_cls.name
        registry = Loss.get_loss_classes()

        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, registry)


class BCETestCase(LossTestCase, unittest.TestCase):
    loss_cls = BinaryCrossEntropy

    def loss_func(self):
        train_size = self.Y.shape[-1]

        preds, epsilon = self.preds, BinaryCrossEntropy.epsilon

        if 1.0 in preds:
            preds = np.clip(preds, epsilon, 1.0 - epsilon)

        positive = self.Y * np.log(preds)
        negative = (1 - self.Y) * np.log(1 - preds)
        loss = np.sum(-(positive + negative)) / train_size
        return np.squeeze(loss)

    def loss_derivatives(self):
        preds, epsilon = self.preds, BinaryCrossEntropy.epsilon

        if 1.0 in preds:
            preds = np.clip(preds, epsilon, 1.0 - epsilon)

        return (1 - self.Y) / (1 - preds) - self.Y / preds

    def test_clipping(self):
        Y = np.random.choice((0, 1), size=(1, 10))
        preds = np.random.choice((0, 1), size=(1, 10))

        self.Y = Y
        self.preds = preds

        expected_loss = self.loss_func()
        loss = self.loss.compute_loss(Y, preds)

        self.assertAlmostEqual(loss, expected_loss)

        expected_grads = self.loss_derivatives()
        grads = self.loss.compute_derivatives(Y, preds)

        np.testing.assert_allclose(grads, expected_grads)


class MSETestCase(LossTestCase, unittest.TestCase):
    loss_cls = MeanSquaredError

    def loss_func(self):
        train_size = self.Y.shape[-1]
        loss = np.sum((self.preds - self.Y) ** 2) / (2 * train_size)
        return np.squeeze(loss)

    def loss_derivatives(self):
        return self.preds - self.Y


class LossRegistryTestCase(unittest.TestCase):
    def test_custom_class_in_registry(self):
        class TestLoss(Loss):
            name = ("test_class", "tc")

            def loss_func(self, labels, preds):
                pass

            def loss_derivative(self, labels, preds):
                pass

        registry = Loss.get_loss_classes()
        self.assertIn("test_class", registry)
        self.assertIn("tc", registry)
