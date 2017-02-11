import unittest
import numpy as np

from layers.layer_fully_connected import LayerFullyConnected
from utils.timing import time_function


class TestLayerFullyConnected(unittest.TestCase):
    def setUp(self):
        self.num_classes = 100
        self.num_points = 1000
        self.point_size = 20000
        self.weights = np.random.randn(self.num_classes, self.point_size)
        self.points = np.random.randn(self.num_points, self.point_size)

    def test_timing(self):
        layer = LayerFullyConnected(self.num_points, self.num_classes, self.weights)
        time_naive = time_function(layer.forward_naive, self.points)
        time_vectorized = time_function(layer.forward_vectorized, self.points)
        # the vectorized implementation should become increasingly faster as
        # the data size increases
        self.assertLess(time_vectorized * 5, time_naive)


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
        self.scores = np.array([[15, 20, 16, 18, 17],
                                [21, 32, 19, 26, 27],
                                [14, 30, 15, 21, 23]])
        self.layer = LayerFullyConnected(self.points.shape[1], self.scores.shape[1], self.weights)

    def test_naive_directed(self):
        scores = self.layer.forward_naive(self.points)
        self.assertTrue(np.array_equal(scores, self.scores))

    def test_vectorized_directed(self):
        scores = self.layer.forward_vectorized(self.points)
        self.assertTrue(np.array_equal(scores, self.scores))

    def test_timing(self):
        time_naive = time_function(self.layer.forward_naive, self.points)
        time_vectorized = time_function(self.layer.forward_vectorized, self.points)
        self.assertLess(time_vectorized * 2, time_naive)
