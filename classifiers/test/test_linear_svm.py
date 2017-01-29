import unittest
import numpy as np
from util.data import load_CIFAR10
from classifiers.linear_svm import LinearSVM

from partitioners.partitioner_range_split import PartitionerRangeSplit
from samplers.sampler_random import SamplerRandom
from importers.importer_cifar10 import ImporterCIFAR10
from util.time import time_function

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

    self.classifier = LinearSVM()

    self.classifier.setup(self.train_points)
    self.train_points = self.classifier.normalize(self.train_points)
    self.classifier.normalize(self.validation_points)
    self.classifier.normalize(self.test_points)

  def xtestSVMLossNaive(self):
    W = np.random.randn(3073, 10) * 0.0001
    regularization_strength = 0.00001
    loss, grad = self.classifier.svm_loss_naive(
        W, self.train_points, self.train_labels, regularization_strength)
    self.assertLess(loss, 10)
    self.assertGreater(loss, 8)

  def testSVMLossVectorized(self):
    W = np.random.randn(3073, 10) * 0.0001
    regularization_strength = 0.00001
    loss, grad = self.classifier.svm_loss_vectorized(
        W, self.train_points, self.train_labels, regularization_strength)
    print('loss %f' % (loss,))
    self.assertLess(loss, 10)
    self.assertGreater(loss, 8)

  def testTiming(self):
    W = np.random.randn(3073, 10) * 0.0001
    regularization_strength = 0.00001

    naive_time = \
      time_function(self.classifier.svm_loss_naive, W, self.train_points,
                         self.train_labels, regularization_strength)

    vectorized_time = \
      time_function(self.classifier.svm_loss_vectorized, W, self.train_points,
                         self.train_labels, regularization_strength)

    print(naive_time)
    print(vectorized_time)
