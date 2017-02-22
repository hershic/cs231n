import unittest
import numpy as np

from classifiers.softmax import ClassifierSoftmax
from importers.cifar10 import ImporterCIFAR10
from layers.fully_connected import LayerFullyConnected
from partitioners.range_split import PartitionerRangeSplit
from samplers.random import SamplerRandom

from lib.gradient_check import eval_numerical_gradient

cifar10_dir = 'datasets/cifar-10-batches-py'


class ClassifierSoftmaxGradientTest:
    def testGradientNaive(self):
        self.classifier.forward_naive(self.scores)
        analyticGradient = self.classifier.gradient.copy()
        numericalGradient = eval_numerical_gradient(
            lambda x: self.classifier.forward(x), self.scores)
        self.assertTrue(np.allclose(numericalGradient, analyticGradient))

    def testGradientVectorized(self):
        self.classifier.forward(self.scores)
        analyticGradient = self.classifier.gradient.copy()
        numericalGradient = eval_numerical_gradient(
            lambda x: self.classifier.forward(x), self.scores)
        self.assertTrue(np.allclose(numericalGradient, analyticGradient))


class TestClassifierSoftmax(unittest.TestCase, ClassifierSoftmaxGradientTest):
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
        self.classifier = ClassifierSoftmax(
            (self.train_points.shape[0], self.num_classifications))
        self.classifier.set_batch_labels(self.train_labels)

        self.scores = self.layer.forward(self.train_points)

    def testSoftmaxLossNaive(self):
        loss = self.classifier.forward_naive(self.scores)
        # Kaparthy says we should expect a probablity around $-\log(0.1) =
        # 2.302$. Check for that here
        self.assertLess(loss, 2.5)
        self.assertGreater(loss, 2)

    def testSoftmaxLossVectorized(self):
        loss = self.classifier.forward(self.scores)
        # Kaparthy says we should expect a probablity around $-\log(0.1) =
        # 2.302$. Check for that here.
        self.assertLess(loss, 2.5)
        self.assertGreater(loss, 2)


class TestClassifierSoftmaxDirected0(unittest.TestCase, ClassifierSoftmaxGradientTest):
    def setUp(self):
        self.scores = np.array([[3.2, 5.1, -1.7]])
        self.train_labels = np.array([0])
        self.classifier = ClassifierSoftmax(self.scores.shape)
        self.classifier.set_batch_labels(self.train_labels)

    def testSoftmaxLossNaive(self):
        loss = self.classifier.forward_naive(self.scores)
        self.assertAlmostEqual(loss, 2.04035515)

    def testSoftmaxLossVectorized(self):
        loss = self.classifier.forward(self.scores)
        self.assertAlmostEqual(loss, 2.04035515)


class TestClassifierSoftmaxDirected1(unittest.TestCase, ClassifierSoftmaxGradientTest):
    def setUp(self):
        self.scores = np.array([[-2.85000, 0.86000, 0.28000]])
        self.train_labels = np.array([2])
        self.classifier = ClassifierSoftmax(self.scores.shape)
        self.classifier.set_batch_labels(self.train_labels)

    def testSoftmaxLossNaive(self):
        loss = self.classifier.forward_naive(self.scores)
        self.assertAlmostEqual(loss, 1.04019057)

    def testSoftmaxLossVectorized(self):
        loss = self.classifier.forward(self.scores)
        self.assertAlmostEqual(loss, 1.04019057)


class TestClassifierSoftmaxDirected2(unittest.TestCase, ClassifierSoftmaxGradientTest):
    def setUp(self):
        self.scores = np.array([[-2.85, 0.86, 0.28],
                                [3.2, 5.1, -1.7]])
        self.train_labels = np.array([2, 0])
        self.classifier = ClassifierSoftmax(self.scores.shape)
        self.classifier.set_batch_labels(self.train_labels)

    def testSoftmaxLossNaive(self):
        loss = self.classifier.forward_naive(self.scores)
        self.assertAlmostEqual(loss, (1.04019057 + 2.04035515) / 2)

    def testSoftmaxLossVectorized(self):
        loss = self.classifier.forward(self.scores)
        self.assertAlmostEqual(loss, (1.04019057 + 2.04035515) / 2)


class TestClassifierSoftmaxDirected3(unittest.TestCase, ClassifierSoftmaxGradientTest):
    def setUp(self):
        self.scores = np.array([[15, 21, 14], [5, 2, 4]], dtype=float)
        self.labels = np.array([2, 1])
        self.classifier = ClassifierSoftmax(self.scores.shape)
        self.classifier.set_batch_labels(self.labels)
