import unittest
import numpy as np

from classifiers.classifier_softmax import ClassifierSoftmax
from importers.importer_cifar10 import ImporterCIFAR10
from layers.layer_fully_connected import LayerFullyConnected
from partitioners.partitioner_range_split import PartitionerRangeSplit
from samplers.sampler_random import SamplerRandom

cifar10_dir = 'datasets/cifar-10-batches-py'


class TestClassifierSoftmax(unittest.TestCase):
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
        self.classifier = ClassifierSoftmax(
            (self.num_classifications, self.train_points.shape[0]))

        self.scores = self.layer.forward_vectorized(self.train_points)

    def xtestSoftmaxLossNaive(self):
        loss, grad = self.classifier.softmax_loss_naive(
            self.scores, self.train_labels)
        self.assertLess(loss, 10)
        self.assertGreater(loss, 8)

    def xtestSoftmaxLossVectorized(self):
        loss, grad = self.classifier.softmax_loss_vectorized(
            self.scores, self.train_labels)
        self.assertLess(loss, 10)
        self.assertGreater(loss, 8)


class TestClassifierSoftmaxDirected0(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([[3.2], [5.1], [-1.7]])
        self.train_labels = np.array([0])
        self.classifier = ClassifierSoftmax(self.scores.shape)

    def testSoftmaxLossNaive(self):
        loss, grad = self.classifier.softmax_loss_naive(
            self.scores, self.train_labels)
        self.assertAlmostEqual(loss, 2.04035515)

    def xtestSoftmaxLossVectorized(self):
        loss, grad = self.classifier.softmax_loss_vectorized(
            self.scores, self.train_labels)
        self.assertAlmostEqual(loss, 2.04035515)


class TestClassifierSoftmaxDirected1(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([[-2.85000], [0.86000], [0.28000]])
        self.train_labels = np.array([2])
        self.classifier = ClassifierSoftmax(self.scores.shape)

    def testSoftmaxLossNaive(self):
        loss, grad = self.classifier.softmax_loss_naive(
            self.scores, self.train_labels)
        self.assertAlmostEqual(loss, 1.04019057)

    def xtestSoftmaxLossVectorized(self):
        loss, grad = self.classifier.softmax_loss_vectorized(
            self.scores, self.train_labels)
        self.assertAlmostEqual(loss, 1.04019057)


class TestClassifierSoftmaxGradient(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[1, 3, 2, 1]], dtype=float)
        self.scores = np.array([[15], [21], [14]], dtype=float)
        self.labels = np.array([2])
        self.classifier = ClassifierSoftmax(self.scores.shape)

    def xtestSoftmaxGradient(self):
        (loss, gradient) = self.classifier.softmax_loss_vectorized(
            self.scores, self.labels)

    def testGradient(self):
        (lossOriginal, analyticGradient) = self.classifier.softmax_loss_naive(
            self.scores, self.labels)
        # preserve the gradient
        analyticGradient = np.array(analyticGradient, copy=True)
        h = 1e-5

        print('\n\n\n\n')

        numericalGradient = np.zeros(analyticGradient.shape)
        for row in range(self.scores.shape[0]):
            for col in range(self.scores.shape[1]):
                self.scores[row, col] += h
                (loss, _) = self.classifier.softmax_loss_naive(
                    self.scores, self.labels)
                grad = (loss - lossOriginal) / h
                numericalGradient[row, col] = grad
                self.scores[row, col] -= h

        print('\nlossOriginal')
        print(lossOriginal)
        print('\nloss')
        print(loss)
        print('\nnumericalGradient')
        print(numericalGradient)
        print('\nanalyticGradient')
        print(analyticGradient)

        self.assertTrue(np.all(np.isclose(numericalGradient, analyticGradient)))
