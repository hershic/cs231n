import unittest
import numpy as np

from layers.fully_connected import LayerFullyConnected
from utils.timing import time_function

class TestLayerFullyConnected(unittest.TestCase):
    def setUp(self):
        pass

    def test_naive_randomized(self):
        num_classes = 3
        num_points = 5
        point_size = 4
        weights = np.random.randn(num_classes, point_size)
        points = np.random.randn(num_points, point_size)
        layer = LayerFullyConnected(weights.shape, points.shape)
        scores = layer.forward_naive(weights, points)


class TestLayerFullyConnectedDirected(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([[1, 2, 3, 2],
                                 [2, 4, 2, 3],
                                 [3, 1, 2, 4]])
        self.points = np.array([[1, 3, 2, 1],
                                [3, 3, 1, 4],
                                [1, 2, 3, 1],
                                [2, 3, 2, 2],
                                [2, 3, 1, 3]])
        self.expected = np.array([[15, 21, 14],
                                  [20, 32, 30],
                                  [16, 19, 15],
                                  [18, 26, 21],
                                  [17, 27, 23]])

    def test_naive_directed(self):
        layer = LayerFullyConnected(self.weights.shape, self.points.shape)
        scores = layer.forward_naive(self.weights, self.points)
        self.assertTrue(np.array_equal(scores, self.expected))

    def test_vectorized_directed(self):
        layer = LayerFullyConnected(self.weights.shape, self.points.shape)
        scores = layer.forward_vectorized(self.weights, self.points)
        self.assertTrue(np.array_equal(scores, self.expected))

    def test_timing(self):
        layer = LayerFullyConnected(self.weights.shape, self.points.shape)
        layer = LayerFullyConnected(self.weights.shape, self.points.shape)

        time_naive = time_function(layer.forward_naive, self.weights, self.points)
        time_vectorized = time_function(layer.forward_vectorized, self.weights, self.points)

        self.assertLess(time_vectorized * 2, time_naive)
