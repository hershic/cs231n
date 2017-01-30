import unittest
import numpy as np

from classifiers.linear_svm import LinearSVM
from importers.importer_cifar10 import ImporterCIFAR10
from layers.layer_fully_connected import LayerFullyConnected
from partitioners.partitioner_range_split import PartitionerRangeSplit
from samplers.sampler_random import SamplerRandom
from utils.data import load_CIFAR10
from utils.timing import time_function

cifar10_dir = 'datasets/cifar-10-batches-py'

class TestLinearSVM(unittest.TestCase):
    def setUp(self):
        self.num_train = 5000
        self.num_test = 500
        self.num_validation = 50

        sampler = SamplerRandom()
        partitioner = PartitionerRangeSplit()
        importer = ImporterCIFAR10(cifar10_dir)

        data = importer.import_all()
        data = partitioner.partition(data, .1)

        train_dataset = sampler.sample(data['train'], self.num_train)
        validation_dataset = sampler.sample(data['train'], self.num_validation)
        test_dataset = sampler.sample(data['test'], self.num_test)

        self.train_labels = train_dataset['labels']
        self.validation_labels = validation_dataset['labels']
        self.test_labels = test_dataset['labels']

        self.train_points = np.reshape(train_dataset['points'], (train_dataset['points'].shape[0], -1))
        self.validation_points = np.reshape(validation_dataset['points'], (validation_dataset['points'].shape[0], -1))
        self.test_points = np.reshape(test_dataset['points'], (test_dataset['points'].shape[0], -1))

        # subtract the mean image
        self.train_points -= np.mean(self.train_points, axis=0)

        self.weights = np.random.randn(10, 3072) * 0.0001
        self.classifier = LinearSVM()
        self.layer = LayerFullyConnected(self.weights.shape, self.train_points.shape)

        self.scores = self.layer.forward_vectorized(self.weights, self.train_points)

    def testSVMLossNaive(self):
        regularization_strength = 0.00001
        loss, grad = \
            self.classifier.svm_loss_naive(self.scores, self.train_labels, regularization_strength)
        self.assertLess(loss, 10)
        self.assertGreater(loss, 8)

    def testSVMLossVectorized(self):
        regularization_strength = 0.00001
        loss, grad = self.classifier.svm_loss_vectorized(
            self.scores, self.train_labels, regularization_strength)
        self.assertLess(loss, 10)
        self.assertGreater(loss, 8)

    def testTiming(self):
        regularization_strength = 0.00001
        naive_time = time_function(self.classifier.svm_loss_naive, self.scores,
                                   self.train_labels, regularization_strength)
        vectorized_time = time_function(self.classifier.svm_loss_vectorized, self.scores,
                                        self.train_labels, regularization_strength)
        self.assertLess(vectorized_time * 50, naive_time)


class TestLinearSVMDirected0(unittest.TestCase):
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
        self.labels = np.array([2, 1, 0, 0, 2])

    def testSVMLossVectorized(self):
        classifier = LinearSVM()
        (loss, gradient) = classifier.svm_loss_vectorized(self.scores, self.labels)
        self.assertAlmostEqual(loss, 6.4)


class TestLinearSVMDirected1(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([[3.2, 1.3, 2.2],
                                [5.1, 4.9, 2.5],
                                [-1.7, 2.0, -3.1]])
        self.labels = np.array([0, 1, 2])

    def testSVMLossVectorized(self):
        classifier = LinearSVM()
        (loss, gradient) = classifier.svm_loss_vectorized(self.scores, self.labels)
        self.assertAlmostEqual(loss, 5.2666666666666657)
