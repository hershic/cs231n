import unittest
import numpy as np

from classifiers.linear_svm import LinearSVM
from importers.importer_cifar10 import ImporterCIFAR10
from layers.layer_fully_connected import LayerFullyConnected
from partitioners.partitioner_range_split import PartitionerRangeSplit
from samplers.sampler_random import SamplerRandom
from utils.timing import time_function

cifar10_dir = 'datasets/cifar-10-batches-py'


class TestLinearSVM(unittest.TestCase):
    def setUp(self):
        self.num_train = 5000
        self.num_test = 500
        self.num_validation = 50
        self.num_classifications = 10

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

        self.train_points = np.reshape(
            train_dataset['points'], (train_dataset['points'].shape[0], -1))
        self.validation_points = np.reshape(
            validation_dataset['points'],
            (validation_dataset['points'].shape[0], -1))
        self.test_points = np.reshape(
            test_dataset['points'], (test_dataset['points'].shape[0], -1))

        # subtract the mean image
        self.train_points -= np.mean(self.train_points, axis=0)

        self.layer = LayerFullyConnected(
            self.train_points.shape[1], self.num_classifications)
        self.classifier = LinearSVM(
            (self.num_classifications, self.train_points.shape[0]))

        self.scores = self.layer.forward_vectorized(self.train_points)

    def testSVMLossNaive(self):
        loss, grad = self.classifier.svm_loss_naive(
            self.scores, self.train_labels)
        self.assertLess(loss, 10)
        self.assertGreater(loss, 8)

    def testSVMLossVectorized(self):
        loss, grad = self.classifier.svm_loss_vectorized(
            self.scores, self.train_labels)
        self.assertLess(loss, 10)
        self.assertGreater(loss, 8)

    def testTiming(self):
        naive_time = time_function(
            self.classifier.svm_loss_naive,
            self.scores, self.train_labels)
        vectorized_time = time_function(
            self.classifier.svm_loss_vectorized,
            self.scores, self.train_labels)
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
        self.classifier = LinearSVM(self.scores.shape)

    def testSVMLossVectorized(self):
        (loss, gradient) = self.classifier.svm_loss_vectorized(
            self.scores, self.labels)
        self.assertAlmostEqual(loss, 6.4)


class TestLinearSVMDirected1(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([[3.2, 1.3, 2.2],
                                [5.1, 4.9, 2.5],
                                [-1.7, 2.0, -3.1]])
        self.labels = np.array([0, 1, 2])
        self.classifier = LinearSVM(self.scores.shape)

    def testSVMLossVectorized(self):
        (loss, gradient) = self.classifier.svm_loss_vectorized(
            self.scores, self.labels)
        self.assertAlmostEqual(loss, 5.2666666666666657)


class TestLinearSVMGradient(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[1, 3, 2, 1]], dtype=float)
        self.scores = np.array([[15], [21], [14]], dtype=float)
        self.labels = np.array([2])
        self.classifier = LinearSVM(self.scores.shape)

    def testSVMGradient(self):
        (loss, gradient) = self.classifier.svm_loss_vectorized(
            self.scores, self.labels)

    def testGradient(self):
        (lossOriginal, analyticGradient) = self.classifier.svm_loss_vectorized(
            self.scores, self.labels)
        # preserve the gradient
        analyticGradient = np.array(analyticGradient, copy=True)
        h = 1e-6

        numericalGradient = np.zeros(analyticGradient.shape)
        for row in range(self.scores.shape[0]):
            for col in range(self.scores.shape[1]):
                self.scores[row, col] += h
                (loss, _) = self.classifier.svm_loss_vectorized(
                    self.scores, self.labels)
                grad = (loss - lossOriginal) / h
                numericalGradient[row, col] = grad
                self.scores[row, col] -= h

        self.assertTrue(np.all(np.isclose(numericalGradient, analyticGradient)))
