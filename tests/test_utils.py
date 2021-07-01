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
    class TestActivation(Activation):
        name = "test_class"

        def activation_func(self, ip):
            pass

        def derivative_func(self, ip):
            pass

    def test_invalid_name_error(self):
        with self.assertRaises(ValueError):
            activation_factory(generate_random_name())

    def test_builtin_activations_membership(self):
        registry = Activation.get_activation_classes()
        names = (Sigmoid.name, Tanh.name, ReLU.name, LeakyReLU.name, ELU.name)

        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, registry)

    def test_custom_class_in_registry(self):
        registry = Activation.get_activation_classes()
        self.assertIn(self.TestActivation.name, registry)

    def test_custom_class_init(self):
        obj = activation_factory("test_class")
        self.assertIsInstance(obj, self.TestActivation)


class LossFactoryTestCase(unittest.TestCase):
    class TestLoss(Loss):
        name = ("test_class_single", "tc")

        def loss_func(self, labels, preds):
            pass

        def loss_derivative(self, labels, preds):
            pass

    def test_invalid_name_error(self):
        with self.assertRaises(ValueError):
            loss_factory(generate_random_name())

    def test_custom_class_init(self):
        obj1 = loss_factory(self.TestLoss.name[0])
        obj2 = loss_factory(self.TestLoss.name[1])
        self.assertIsInstance(obj1, self.TestLoss)
        self.assertIsInstance(obj2, self.TestLoss)
