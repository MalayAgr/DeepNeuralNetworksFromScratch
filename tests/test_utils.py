import unittest

from dnn.activations import Activation
from dnn.utils import activation_factory


class ActivationFactoryTestCase(unittest.TestCase):
    class TestActivation(Activation):
        name = "test_class"

        def activation_func(self, ip):
            pass

        def derivative_func(self, ip):
            pass

    def test_custom_class_in_registry(self):
        registry = Activation._get_activation_classes()
        self.assertIn("test_class", registry)

    def test_custom_class_instantiation(self):
        obj = activation_factory("test_class")
        self.assertIsInstance(obj, self.TestActivation)
