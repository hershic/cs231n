import unittest
import numpy as np

from layers.relu import LayerReLU
from lib.gradient_check import eval_numerical_gradient_array


class TestLayerRELUDirected0(unittest.TestCase):
    def setUp(self):
        self.relu = LayerReLU()
        self.input = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        self.activations = np.array([[0,          0,          0,          0         ],
                                     [0,          0,          0.04545455, 0.13636364],
                                     [0.22727273, 0.31818182, 0.40909091, 0.5       ]])
        self.gradient_in = np.random.randn(*self.input.shape)

    def testReLU(self):
        activations = self.relu.forward(self.input)
        self.assertTrue(np.allclose(self.activations, activations))

    def testGradient(self):
        _ = self.relu.forward(self.input)
        analyticGradient = self.relu.backward(self.gradient_in).copy()
        numericalGradient = eval_numerical_gradient_array(
            lambda inputs: self.relu.forward(inputs), self.input, self.gradient_in)

        self.assertTrue(np.allclose(analyticGradient, numericalGradient))
