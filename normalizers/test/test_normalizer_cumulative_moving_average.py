import unittest
import numpy as np

from importers.importer_cifar10 import ImporterCIFAR10
from partitioners.partitioner_range_split import PartitionerRangeSplit
from samplers.sampler_random import SamplerRandom
from normalizers.normalizer_cumulative_moving_average import NormalizerCumulativeMovingAverage

cifar10_dir = 'datasets/cifar-10-batches-py'


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


class TestCumulativeMovingAverageOnCIFAR10(unittest.TestCase):
    def setUp(self):
        self.num_train = 5000

        sampler = SamplerRandom()
        partitioner = PartitionerRangeSplit()
        importer = ImporterCIFAR10(cifar10_dir)

        data = importer.import_all()
        data = partitioner.partition(data, .1)

        train_dataset = sampler.sample(data['train'], self.num_train)

        self.train_points = np.reshape(train_dataset['points'],
                                       (train_dataset['points'].shape[0], -1))

    def test_mean(self):
        np_mean = np.mean(self.train_points, axis=0).reshape(-1, 1)

        normalizer = NormalizerCumulativeMovingAverage(self.train_points.shape[1])
        normalizer.calculate_batch(self.train_points)
        normalized_mean = normalizer.mean.reshape(-1, 1)

        subtracted = np.absolute(np_mean - normalized_mean)

        self.assertAlmostEqual(np.max(subtracted), 0)
        self.assertAlmostEqual(np.min(subtracted), 0)
        self.assertAlmostEqual(np.mean(subtracted), 0)
