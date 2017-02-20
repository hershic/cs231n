import unittest
import numpy as np

from classifiers.classifier_softmax import ClassifierSoftmax
from importers.importer_cifar10 import ImporterCIFAR10
from layers.layer_fully_connected import LayerFullyConnected
from partitioners.partitioner_range_split import PartitionerRangeSplit
from samplers.sampler_random import SamplerRandom

cifar10_dir = 'datasets/cifar-10-batches-py'

# TODO: clean this up! There's so much duplicated code below...


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
            (self.train_points.shape[1], self.num_classifications))
        self.layer.bias = np.zeros(self.num_classifications)
        self.layer.weights = np.zeros((self.train_points.shape[1], self.num_classifications))
        self.classifier = ClassifierSoftmax(
            (self.train_points.shape[0], self.num_classifications))

        self.scores = self.layer.forward(self.train_points)

    def testSoftmaxLossNaive(self):
        loss, grad = self.classifier.forward_naive(
            self.scores, self.train_labels)
        # Kaparthy says we should expect a probablity around $-\log(0.1) =
        # 2.302$. Check for that here
        self.assertLess(loss, 2.5)
        self.assertGreater(loss, 2)

    def testSoftmaxLossVectorized(self):
        loss, grad = self.classifier.forward(
            self.scores, self.train_labels)
        # Kaparthy says we should expect a probablity around $-\log(0.1) =
        # 2.302$. Check for that here.
        self.assertLess(loss, 2.5)
        self.assertGreater(loss, 2)


class TestClassifierSoftmaxDirected0(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([[3.2, 5.1, -1.7]])
        self.train_labels = np.array([0])
        self.classifier = ClassifierSoftmax(self.scores.shape)

    def testSoftmaxLossNaive(self):
        loss, grad = self.classifier.forward_naive(
            self.scores, self.train_labels)
        self.assertAlmostEqual(loss, 2.04035515)

    def testSoftmaxLossVectorized(self):
        loss, grad = self.classifier.forward(
            self.scores, self.train_labels)
        self.assertAlmostEqual(loss, 2.04035515)


class TestClassifierSoftmaxDirected1(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([[-2.85000, 0.86000, 0.28000]])
        self.train_labels = np.array([2])
        self.classifier = ClassifierSoftmax(self.scores.shape)

    def testSoftmaxLossNaive(self):
        loss, grad = self.classifier.forward_naive(
            self.scores, self.train_labels)
        self.assertAlmostEqual(loss, 1.04019057)

    def testSoftmaxLossVectorized(self):
        loss, grad = self.classifier.forward(
            self.scores, self.train_labels)
        self.assertAlmostEqual(loss, 1.04019057)


class TestClassifierSoftmaxDirected2(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([[-2.85, 0.86, 0.28],
                                [3.2, 5.1, -1.7]])
        self.train_labels = np.array([2, 0])
        self.classifier = ClassifierSoftmax(self.scores.shape)

    def testSoftmaxLossNaive(self):
        loss, grad = self.classifier.forward_naive(
            self.scores, self.train_labels)
        self.assertAlmostEqual(loss, (1.04019057 + 2.04035515) / 2)

    def testSoftmaxLossVectorized(self):
        loss, grad = self.classifier.forward(
            self.scores, self.train_labels)
        self.assertAlmostEqual(loss, (1.04019057 + 2.04035515) / 2)


class TestClassifierSoftmaxGradient(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[1, 3, 2, 1]], dtype=float)
        self.scores = np.array([[15, 21, 14]], dtype=float)
        self.labels = np.array([2])
        self.classifier = ClassifierSoftmax(self.scores.shape)

    def testGradientNaive(self):
        (loss, gradient) = self.classifier.forward_naive(
            self.scores, self.labels)
        self.gradientCheck(loss, gradient)

    def testGradientVectorized(self):
        (loss, gradient) = self.classifier.forward(
            self.scores, self.labels)
        self.gradientCheck(loss, gradient)

    def gradientCheck(self, lossOriginal, analyticGradient):
        # preserve the gradient
        analyticGradient = np.array(analyticGradient, copy=True)
        h = 1e-5

        numericalGradient = np.zeros(analyticGradient.shape)
        for row in range(self.scores.shape[0]):
            for col in range(self.scores.shape[1]):
                self.scores[row, col] += h
                (loss, _) = self.classifier.forward_naive(
                    self.scores, self.labels)
                grad = (loss - lossOriginal) / h
                numericalGradient[row, col] = grad
                self.scores[row, col] -= h
        self.assertTrue(np.all(np.isclose(numericalGradient, analyticGradient)))
