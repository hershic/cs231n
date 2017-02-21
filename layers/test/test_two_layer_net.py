import unittest
import numpy as np

from classifiers.softmax import ClassifierSoftmax
from layers.relu import LayerReLU
from layers.fully_connected import LayerFullyConnected
from regularizers.l2 import RegularizerL2

from lib.gradient_check import eval_numerical_gradient
from utils.compose import compose


class TestTwoLayerNetFixture(unittest.TestCase):
    def initConstants(self):
        pass

    def initNetwork(self):
        self.forward_input = np.random.randn(self.num_points, self.point_size)
        self.forward_classifications = np.random.randint(self.num_classifications,
                                                         size=self.num_points)
        self.backward_input = np.random.randn(self.num_points, self.num_classifications)

        self.layer0 = LayerFullyConnected((self.point_size, self.hidden_layer_size))
        self.layer0_activations = LayerReLU()
        self.layer1 = LayerFullyConnected((self.hidden_layer_size, self.num_classifications))
        self.layer1_activations = LayerReLU()
        self.classifier = ClassifierSoftmax((self.num_points, self.num_classifications))

        self.regularizer = RegularizerL2()
        self.regularizer.addLayer(self.layer0)
        self.regularizer.addLayer(self.layer1)

        self.forward = compose(self.layer0.forward, self.layer0_activations.forward,
                               self.layer1.forward, self.layer1_activations.forward,
                               self.classifier.forward)
        self.backward = compose(self.classifier.backward,
                                self.layer1_activations.backward, self.layer1.backward,
                                self.layer0_activations.backward, self.layer0.backward)

        self.classifier.set_batch_labels(self.forward_classifications)

    def setUp(self):
        self.initConstants()
        self.initNetwork()


class TwoLayerNetGradientTest:
    def test_gradient(self):
        for layer in [self.layer0, self.layer1]:
            numerical_gradient = eval_numerical_gradient(
                lambda _: self.forward(self.forward_input), layer.weights)
            self.backward(1)
            self.assertTrue(np.allclose(numerical_gradient, layer.d_weights))

        for layer in [self.layer0, self.layer1]:
            numerical_gradient = eval_numerical_gradient(
                lambda _: np.sum(self.forward(self.forward_input)), layer.bias)
            self.backward(1)
            self.assertTrue(np.allclose(numerical_gradient, layer.d_bias))


class TestTwoLayerNetDirected0(TestTwoLayerNetFixture, TwoLayerNetGradientTest):
    def initConstants(self):
        self.num_points = 3
        self.point_size = 5
        self.num_classifications = 7
        # self.hidden_layer_size = 50
        self.hidden_layer_size = 5

    def initNetwork(self):
        super().initNetwork()

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

        self.classifier.set_batch_labels(np.array([0, 5, 1]))

    def xtest_forward_fc_only(self):
        self.forward = compose(self.layer0.forward, self.layer0_activations.forward,
                               self.layer1.forward, self.layer1_activations.forward)

        correct_scores = np.array(
            [[11.53165108, 12.2917344, 13.05181771, 13.81190102,
              14.57198434, 15.33206765, 16.09215096],
             [12.05769098, 12.74614105, 13.43459113, 14.1230412,
              14.81149128, 15.49994135, 16.18839143],
             [12.58373087, 13.20054771, 13.81736455, 14.43418138,
              15.05099822, 15.66781506, 16.2846319]])

        scores = self.forward(self.forward_input)
        self.assertTrue(np.allclose(scores, correct_scores))

    def xtest_forward(self):
        self.classifier.set_batch_labels(np.array([0, 5, 1]))
        self.assertAlmostEqual(self.forward(self.forward_input), 3.4702243556)

        self.assertAlmostEqual(26.5948426952,
                               self.forward(self.forward_input) +
                               self.regularizer.calculate())


class TestTwoLayerNetDirected1(TestTwoLayerNetFixture, TwoLayerNetGradientTest):
    def initConstants(self):
        self.num_points = 10
        self.point_size = 50
        self.num_classifications = 10
        self.hidden_layer_size = 20

    def test_initialization(self):
        standard_deviation = 1e-2

        standard_deviation_layer0_weights = abs(self.layer0.weights.std() - standard_deviation)
        standard_deviation_layer1_weights = abs(self.layer1.weights.std() - standard_deviation)

        self.assertLess(standard_deviation_layer0_weights, standard_deviation / 10)
        self.assertLess(standard_deviation_layer1_weights, standard_deviation / 10)
