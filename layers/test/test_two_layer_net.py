import unittest
import numpy as np

from classifiers.softmax import ClassifierSoftmax
from layers.relu import LayerReLU
from layers.fully_connected import LayerFullyConnected

from lib.gradient_check import eval_numerical_gradient_array
from utils.compose import compose


class TestTwoLayerNet(unittest.TestCase):
    def setUp(self):
        self.num_points = 3
        self.point_size = 5
        self.num_classifications = 7
        self.hidden_layer_size = 50

        self.forward_input = np.random.randn(self.num_points, self.point_size)
        self.forward_classifications = np.random.randint(self.num_classifications,
                                                         size=self.num_points)
        self.backward_input = np.random.randn(self.num_points, self.num_classifications)

        # first layer, compute 10 classifications per point
        self.layer0 = LayerFullyConnected((self.point_size, self.hidden_layer_size))
        self.layer0_activations = LayerReLU()
        self.layer1 = LayerFullyConnected((self.hidden_layer_size, self.num_classifications))
        self.layer1_activations = LayerReLU()
        self.classifier = ClassifierSoftmax((self.num_points, self.num_classifications))

        self.forward = compose(self.layer0.forward, self.layer0_activations.forward,
                               self.layer1.forward, self.layer1_activations.forward,
                               self.classifier.forward)
        self.backward = compose(self.classifier.backward,
                                self.layer1_activations.backward, self.layer1.backward,
                                self.layer0_activations.backward, self.layer0.backward)

        self.classifier.set_batch_labels(self.forward_classifications)

    def test_initialization(self):
        standard_deviation = 1e-2

        standard_deviation_layer0_weights = abs(self.layer0.weights.std() - standard_deviation)
        standard_deviation_layer1_weights = abs(self.layer1.weights.std() - standard_deviation)

        self.assertLess(standard_deviation_layer0_weights, standard_deviation / 10)
        self.assertLess(standard_deviation_layer1_weights, standard_deviation / 10)

    def test_forward(self):
        self.layer0.weights = np.linspace(
            -0.7, 0.3, num=self.point_size * self.hidden_layer_size) \
            .reshape(self.point_size, self.hidden_layer_size)
        self.layer0.bias = np.linspace(-0.1, 0.9, num=self.hidden_layer_size)
        self.layer1.weights = np.linspace(
            -0.3, 0.4, num=self.hidden_layer_size * self.num_classifications) \
            .reshape(self.hidden_layer_size, self.num_classifications)
        self.layer1.bias = np.linspace(-0.9, 0.1, num=self.num_classifications)

        self.forward_input = np.linspace(
            -5.5, 4.5, num=self.num_points * self.point_size) \
            .reshape(self.point_size, self.num_points) \
            .T

        correct_scores = np.asarray(
            [[11.53165108, 12.2917344, 13.05181771, 13.81190102,
              14.57198434, 15.33206765, 16.09215096],
             [12.05769098, 12.74614105, 13.43459113, 14.1230412,
              14.81149128, 15.49994135, 16.18839143],
             [12.58373087, 13.20054771, 13.81736455, 14.43418138,
              15.05099822, 15.66781506, 16.2846319]])

        forward = compose(self.layer0.forward, self.layer0_activations.forward,
                          self.layer1.forward, self.layer1_activations.forward)

        scores = forward(self.forward_input)
        self.assertTrue(np.allclose(scores, correct_scores))

        self.classifier.set_batch_labels(np.array([0, 5, 1]))
        self.assertAlmostEqual(self.forward(self.forward_input), 3.4702243556)
