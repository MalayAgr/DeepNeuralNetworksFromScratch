import random
import string
import unittest

import numpy as np
from dnn.activations import Activation
from dnn.loss import Loss
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

    def test_custom_class_in_registry(self):
        registry = Activation._get_activation_classes()
        self.assertIn(self.TestActivation.name, registry)

    def test_custom_class_init(self):
        obj = activation_factory("test_class")
        self.assertIsInstance(obj, self.TestActivation)


class LossFactoryTestCase(unittest.TestCase):
    class TestLossSingleName(Loss):
        name = "test_class_single"

        def loss_func(self, preds):
            pass

        def loss_derivative(self, preds):
            pass

    class TestLossMultiName(Loss):
        name = ("test_class_multi", "tc")

        def loss_func(self, preds):
            pass

        def loss_derivative(self, preds):
            pass

    def test_invalid_name_error(self):
        with self.assertRaises(ValueError):
            loss_factory(generate_random_name(), Y=np.zeros(shape=(1, 1)))

    def test_custom_class_in_registry(self):
        registry = Loss._get_loss_classes()
        self.assertIn(self.TestLossSingleName.name, registry)

    def test_custom_class_init(self):
        Y = np.random.randn(2, 3)
        obj = loss_factory(self.TestLossSingleName.name, Y=Y)
        self.assertIsInstance(obj, self.TestLossSingleName)

    def test_custom_class_in_registry_multi(self):
        registry = Loss._get_loss_classes()
        self.assertIn(self.TestLossMultiName.name[0], registry)
        self.assertIn(self.TestLossMultiName.name[1], registry)

    def test_custom_class_init_multi(self):
        Y = np.random.randn(2, 3)
        obj1 = loss_factory(self.TestLossMultiName.name[0], Y=Y)
        obj2 = loss_factory(self.TestLossMultiName.name[1], Y=Y)
        self.assertIsInstance(obj1, self.TestLossMultiName)
        self.assertIsInstance(obj2, self.TestLossMultiName)
