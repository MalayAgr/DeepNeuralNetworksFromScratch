from abc import abstractmethod
from dnn.loss import BinaryCrossEntropy, MeanSquaredError
from dnn.activations import Sigmoid

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

        self.loss = self.loss_cls(Y=Y)

        self.Y = Y
        self.preds = preds

    def test_validate_input(self):
        preds = np.random.randn(1, 6)

        with self.assertRaises(AttributeError):
            self.loss.validate_input(preds)

    def test_init(self):
        self.assertEqual(self.loss.labels.shape, self.Y.shape)
        self.assertEqual(self.loss.train_size, self.Y.shape[-1])

    def test_compute_loss(self):
        expected_loss = self.loss_func()
        self.assertAlmostEqual(self.loss.compute_loss(self.preds), expected_loss)

    def test_compute_derivatives(self):
        expected_derivatives = self.loss_derivatives()

        derivatives = self.loss.compute_derivatives(self.preds)

        self.assertEqual(derivatives.shape, expected_derivatives.shape)
        np.testing.assert_allclose(derivatives, expected_derivatives)


class BSETestCase(LossTestCase, unittest.TestCase):
    loss_cls = BinaryCrossEntropy

    def loss_func(self):
        train_size = self.Y.shape[-1]
        positive = self.Y * np.log(self.preds)
        negative = (1 - self.Y) * np.log(1 - self.preds)
        loss = np.sum(-(positive + negative)) / train_size
        return np.squeeze(loss)

    def loss_derivatives(self):
        return (1 - self.Y) / (1 - self.preds) - self.Y / self.preds


class MSETestCase(LossTestCase, unittest.TestCase):
    loss_cls = MeanSquaredError

    def loss_func(self):
        train_size = self.Y.shape[-1]
        loss = np.sum((self.preds - self.Y) ** 2) / (2 * train_size)
        return np.squeeze(loss)

    def loss_derivatives(self):
        train_size = self.Y.shape[-1]
        return (self.preds - self.Y) / train_size
