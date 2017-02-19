import unittest
import numpy as np

from layers.relu import LayerReLU
from layers.layer_fully_connected import LayerFullyConnected

from lib.gradient_check import eval_numerical_gradient_array
from utils.compose import compose

class TestLayerReLUDirected0(unittest.TestCase):
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
        analyticGradient = self.relu.backward(self.gradient_in)
        numericalGradient = eval_numerical_gradient_array(
            lambda inputs: self.relu.forward(inputs), self.input, self.gradient_in)

        self.assertTrue(np.allclose(analyticGradient, numericalGradient))


class TestLayerFullyConnectedWithLayerReLU(unittest.TestCase):
    def setUp(self):

        num_images = 50
        image_size = 100
        num_classifications = 10

        self.forward_input = np.random.randn(num_images, image_size)
        self.backward_input = np.random.randn(num_images, num_classifications)

        # first layer, compute 10 classifications per image
        self.layer0 = LayerFullyConnected((image_size, num_classifications))
        self.layer1 = LayerReLU()

    def testGradientsFullyConnectedWithReLU(self):
        run_layers_forward = compose(self.layer0.forward_vectorized, self.layer1.forward)
        run_layers_backward = compose(self.layer1.backward, self.layer0.backward_vectorized)

        # for side-effects
        _ = run_layers_forward(self.forward_input)
        layer0_backward_analytic = run_layers_backward(self.backward_input)

        layer0_backward_numerical = eval_numerical_gradient_array(
            lambda forward_input: run_layers_forward(forward_input),
            self.forward_input, self.backward_input)

        self.assertTrue(np.allclose(layer0_backward_analytic, layer0_backward_numerical))
