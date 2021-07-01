import random
import string
import unittest

import numpy as np
from dnn.activations import ELU, Activation, LeakyReLU, ReLU, Sigmoid, Tanh
from dnn.loss import Loss, BinaryCrossEntropy, MeanSquaredError
from dnn.utils import activation_factory, loss_factory


def generate_random_name():
    return "".join(random.choices(string.ascii_lowercase + string.ascii_uppercase))


class ActivationFactoryTestCase(unittest.TestCase):
    def test_invalid_name_error(self):
        with self.assertRaises(ValueError):
            activation_factory(generate_random_name())

    def test_custom_class_init(self):
        class TestActivation(Activation):
            name = "test_class"

            def activation_func(self, ip):
                pass

            def derivative_func(self, ip):
                pass

        obj = activation_factory("test_class")
        self.assertIsInstance(obj, TestActivation)


class LossFactoryTestCase(unittest.TestCase):
    def test_invalid_name_error(self):
        with self.assertRaises(ValueError):
            loss_factory(generate_random_name())

    def test_custom_class_init(self):
        class TestLoss(Loss):
            name = ("test_class", "tc")

            def loss_func(self, labels, preds):
                pass

            def loss_derivative(self, labels, preds):
                pass

        obj1 = loss_factory("test_class")
        obj2 = loss_factory("tc")
        self.assertIsInstance(obj1, TestLoss)
        self.assertIsInstance(obj2, TestLoss)
