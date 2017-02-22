import unittest
import numpy as np

from classifiers.svm import ClassifierSVM
from importers.cifar10 import ImporterCIFAR10
from layers.fully_connected import LayerFullyConnected
from lib.gradient_check import eval_numerical_gradient
from partitioners.range_split import PartitionerRangeSplit
from samplers.random import SamplerRandom

from utils.timing import time_function
from utils.allow_failure import allow_failure

cifar10_dir = 'datasets/cifar-10-batches-py'


class TestLinearSVM(unittest.TestCase):
    def setUp(self):
        self.num_train = 500
        self.num_test = 50
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
            (self.train_points.shape[1], self.num_classifications))
        self.layer.bias = np.zeros(self.num_classifications)
        self.layer.weights = np.zeros((self.train_points.shape[1], self.num_classifications))
        self.classifier = ClassifierSVM(
            (self.train_points.shape[0], self.num_classifications))
        self.classifier.set_batch_labels(self.train_labels)
        self.scores = self.layer.forward(self.train_points)

    def testSVMLossNaive(self):
        loss = self.classifier.forward_naive(self.scores)
        self.assertLess(loss, 10)
        self.assertGreater(loss, 8)
        self.classifier.set_batch_labels(self.train_labels)

    def testSVMLossVectorized(self):
        loss = self.classifier.forward(self.scores)
        self.assertLess(loss, 10)
        self.assertGreater(loss, 8)

    @allow_failure
    def testTiming(self):
        naive_time = time_function(self.classifier.forward_naive, self.scores)
        vectorized_time = time_function(self.classifier.forward, self.scores)
        self.assertLess(vectorized_time * 10, naive_time)


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
        self.scores = np.array([[15, 21, 14],
                                [20, 32, 30],
                                [16, 19, 15],
                                [18, 26, 21],
                                [17, 27, 23]])
        self.labels = np.array([2, 1, 0, 0, 2])
        self.classifier = ClassifierSVM(self.scores.shape)
        self.classifier.set_batch_labels(self.labels)

    def testSVMLossVectorized(self):
        loss = self.classifier.forward(self.scores)
        self.assertAlmostEqual(loss, 6.4)


class TestLinearSVMDirected1(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([[3.2, 5.1, -1.7],
                                [1.3, 4.9, 2],
                                [2.2, 2.5, -3.1]])
        self.labels = np.array([0, 1, 2])
        self.classifier = ClassifierSVM(self.scores.shape)
        self.classifier.set_batch_labels(self.labels)

    def testSVMLossVectorized(self):
        loss = self.classifier.forward(self.scores)
        self.assertAlmostEqual(loss, 5.2666666666666657)


class TestLinearSVMGradient(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[1, 3, 2, 1]], dtype=float)
        self.scores = np.array([[15, 21, 14]], dtype=float)
        self.gradient = np.array([[1, 1, -2]], dtype=float)
        self.labels = np.array([2])
        self.classifier = ClassifierSVM(self.scores.shape)
        self.classifier.set_batch_labels(self.labels)

    def testGradientVectorized(self):
        self.classifier.forward(self.scores)
        analyticGradient = self.classifier.gradient.copy()
        numericalGradient = eval_numerical_gradient(
            lambda scores: self.classifier.forward(scores), self.scores)
        self.assertTrue(np.all(np.isclose(numericalGradient, analyticGradient)))

    def testGradientNaive(self):
        self.classifier.forward_naive(self.scores)
        analyticGradient = self.classifier.gradient.copy()
        numericalGradient = eval_numerical_gradient(
            lambda scores: self.classifier.forward_naive(scores), self.scores)
        self.assertTrue(np.all(np.isclose(numericalGradient, analyticGradient)))
