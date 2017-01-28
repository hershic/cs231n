import unittest
import numpy as np
from util.data import load_CIFAR10
from classifiers.linear_svm import LinearSVM

from partitioners.partitioner_range_split import PartitionerRangeSplit
from samplers.sampler_random import SamplerRandom
from importers.importer_cifar10 import ImporterCIFAR10

cifar10_dir = 'datasets/cifar-10-batches-py'


class TestLinearSVM(unittest.TestCase):
  def setUp(self):
    self.num_train = 500
    self.num_test = 50

    sampler = SamplerRandom()
    partitioner = PartitionerRangeSplit()
    importer = ImporterCIFAR10(cifar10_dir)

    data = importer.import_all()
    data = partitioner.partition(data, .1)

    self.train_points = sampler.sample(data['train']['points'], self.num_train)
    self.train_labels = sampler.sample(data['train']['labels'], self.num_train)
    self.test_points = sampler.sample(data['test']['points'], self.num_test)
    self.test_labels = sampler.sample(data['test']['labels'], self.num_test)

    self.train_points = np.reshape(self.train_points, (self.train_points.shape[0], -1))
    self.test_points = np.reshape(self.test_points, (self.test_points.shape[0], -1))

    # X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    # X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    # X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    # X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    # data = partitioner.partition(data, 6, 0)

    # self.train_points = sampler.sample(data['train']['points'], self.num_train)
    # self.train_labels = sampler.sample(data['train']['labels'], self.num_train)
    # self.test_points = sampler.sample(data['test']['points'], self.num_test)
    # self.test_labels = sampler.sample(data['test']['labels'], self.num_test)


    # self.classifier = KNearestNeighbor()
    # self.classifier.train(self.train_points, self.train_labels)

  def testSomething(self):
    pass
