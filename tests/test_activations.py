from abc import abstractmethod
import unittest

import numpy as np
from dnn.activations import LeakyReLU, ReLU, Sigmoid, Tanh


class BaseActivationTestCase:
    """
    Test an activation function class class
    """

    act_cls = None

    @abstractmethod
    def activations(self, ip):
        pass

    @abstractmethod
    def derivatives(self, ip):
        pass

    def setUp(self):
        self.act_obj = self.act_cls()

    def test_no_input(self):
        """
        Test error when no input is supplied
        """
        with self.assertRaises(AttributeError):
            self.act_obj.calculate_activations()

        with self.assertRaises(AttributeError):
            self.act_obj.calculate_derivatives()

    def test_custom_input(self):
        """
        Test with explicit input passed to calculate_activations() and calculate_derivatives()
        """
        ip = np.random.randn(5, 1)

        activations = self.activations(ip)
        derivatives = self.derivatives(activations)

        np.testing.assert_allclose(self.act_obj.calculate_activations(ip), activations)

        np.testing.assert_allclose(self.act_obj.calculate_derivatives(ip), derivatives)

    def test_init_input(self):
        """
        Test input supplied during initialization
        """
        ip = np.random.randn(5, 1)

        activations = self.activations(ip)
        derivatives = self.derivatives(activations)

        self.act_obj.ip = ip

        np.testing.assert_allclose(self.act_obj.calculate_activations(), activations)

        np.testing.assert_allclose(self.act_obj.activations, activations)

        np.testing.assert_allclose(self.act_obj.calculate_derivatives(), derivatives)

        np.testing.assert_allclose(self.act_obj.derivatives, derivatives)

    def test_custom_input_override(self):
        """
        Test explicit input overriding the input supplied during init
        """
        custom_ip = np.random.randn(5, 1)

        activations = self.activations(custom_ip)
        derivatives = self.derivatives(activations)

        init_ip = np.random.randn(5, 1)
        self.act_obj.ip = init_ip

        np.testing.assert_allclose(
            self.act_obj.calculate_activations(custom_ip), activations
        )

        self.assertIsNone(self.act_obj.activations)

        np.testing.assert_allclose(
            self.act_obj.calculate_derivatives(custom_ip), derivatives
        )

        self.assertIsNone(self.act_obj.derivatives)


class SigmoidTestCase(BaseActivationTestCase, unittest.TestCase):
    act_cls = Sigmoid

    def activations(self, ip):
        return 1 / (1 + np.exp(-ip))

    def derivatives(self, ip):
        return ip * (1 - ip)


class TanhTestCase(BaseActivationTestCase, unittest.TestCase):
    act_cls = Tanh

    def activations(self, ip):
        return np.tanh(ip)

    def derivatives(self, ip):
        return 1 - ip ** 2


class ReLUTestCase(BaseActivationTestCase, unittest.TestCase):
    act_cls = ReLU

    def activations(self, ip):
        return np.maximum(0, ip)

    def derivatives(self, ip):
        return np.where(ip > 0, 1.0, 0.0)


class LeakyReLUTestCase(BaseActivationTestCase, unittest.TestCase):
    act_cls = LeakyReLU

    def test_default_alpha(self):
        self.assertAlmostEqual(self.act_obj.alpha, 0.01)

    def test_custom_alpha(self):
        obj_custom_alpha = self.act_cls(alpha=0.02)
        self.assertAlmostEqual(obj_custom_alpha.alpha, 0.02)

    def activations(self, ip):
        return np.maximum(0.01 * ip, ip)

    def derivatives(self, ip):
        return np.where(ip > 0, 1.0, 0.01)
