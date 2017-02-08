import unittest
import numpy as np

from normalizers.normalizer_cumulative_moving_average import NormalizerCumulativeMovingAverage


class TestCumulativeMovingAverage(unittest.TestCase):
    def setUp(self):
        self.normalizer = NormalizerCumulativeMovingAverage((1,))

    def testBasicMovingAverage(self):
        numbers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.calculate_batch(numbers)
        self.assertAlmostEqual(self.normalizer.mean, 3.0)
        self.normalize_batch(numbers)
        self.assertAlmostEqual(np.sum(numbers), 0)

    def calculate_batch(self, numbers, shape=(-1, 1)):
        numbers = np.reshape(numbers, shape)
        for number in numbers:
            self.normalizer.calculate_batch(number)

    def normalize_batch(self, numbers, shape=(-1, 1)):
        numbers = np.reshape(numbers, shape)
        for number in numbers:
            self.normalizer.normalize_batch(number)
