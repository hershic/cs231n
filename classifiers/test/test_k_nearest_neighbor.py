import unittest
import numpy as np

from classifiers import KNearestNeighbor
from utils.data import load_CIFAR10
from utils.timing import time_function

from partitioners.partitioner_k_folds import PartitionerKFolds
from samplers.sampler_range_mask import SamplerRangeMask
from importers.importer_cifar10 import ImporterCIFAR10

from utils.allow_failure import allow_failure

cifar10_dir = 'datasets/cifar-10-batches-py'


class TestKNearestNeighbor(unittest.TestCase):
    def setUp(self):
        self.num_train = 500
        self.num_test = 50

        sampler = SamplerRangeMask()
        partitioner = PartitionerKFolds()
        importer = ImporterCIFAR10(cifar10_dir)

        data = importer.import_all()
        data = partitioner.partition(data, 6, 0)

        train_dataset = sampler.sample(data['train'], self.num_train)
        test_dataset = sampler.sample(data['test'], self.num_test)

        self.train_labels = train_dataset['labels']
        self.test_labels = test_dataset['labels']

        self.train_points = np.reshape(train_dataset['points'], (train_dataset['points'].shape[0], -1))
        self.test_points = np.reshape(test_dataset['points'], (test_dataset['points'].shape[0], -1))

        self.classifier = KNearestNeighbor()
        self.classifier.train(self.train_points, self.train_labels)

    def test_dist_two_loops(self):
        dists_two = self.classifier._compute_distances_two_loops(self.test_points)

        # Now implement the function predict_labels and run the code below: We use
        # k = 1 (which is Nearest Neighbor).
        test_labels_pred = self.classifier._predict_labels(dists_two, k=5)

        # Compute and print the fraction of correctly predicted examples

        # accuracy should be somewhere around 15-25% depending on the subsampling
        # pattern
        num_correct = np.sum(test_labels_pred == self.test_labels)
        accuracy = float(num_correct) / self.num_test
        self.assertGreater(num_correct, 1)
        self.assertLess(accuracy, 0.3)
        self.assertGreater(accuracy, 0.1)

    def test_dist_one_loop(self):
        # Now lets speed up distance matrix computation by using partial
        # vectorization with one loop. Implement the function
        # compute_distances_one_loop and run the code below:
        dists_two = self.classifier._compute_distances_two_loops(self.test_points)
        dists_one = self.classifier._compute_distances_one_loop(self.test_points)

        # To ensure that our vectorized implementation is correct, we make sure
        # that it agrees with the naive implementation. There are many ways to
        # decide whether two matrices are similar; one of the simplest is the
        # Frobenius norm. In case you haven't seen it before, the Frobenius norm of
        # two matrices is the square root of the squared sum of differences of all
        # elements; in other words, reshape the matrices into vectors and compute
        # the Euclidean distance between them.
        difference = np.linalg.norm(dists_two - dists_one, ord='fro')
        self.assertAlmostEqual(difference, 0.0)

    def test_dist_no_loops(self):
        # Now implement the fully vectorized version inside
        # compute_distances_no_loops and run the code
        dists_one = self.classifier._compute_distances_one_loop(self.test_points)
        dists_none = self.classifier._compute_distances_no_loops(self.test_points)

        # check that the distance matrix agrees with the one we computed before:
        difference = np.linalg.norm(dists_one - dists_none, ord='fro')
        self.assertAlmostEqual(difference, 0.0)

    # Ensure the algorithms are sane
    @allow_failure
    def test_timing(self):
        two_loop_time = time_function(self.classifier._compute_distances_two_loops, self.test_points)
        one_loop_time = time_function(self.classifier._compute_distances_one_loop, self.test_points)
        no_loop_time = time_function(self.classifier._compute_distances_no_loops, self.test_points)

        # The vectorized no-loops version of the distance computatino should be
        # about 10x faster than the 1-loop version; the 1-loop version should be of
        # similar speed (but slightly faster) compared to the 2-loop version
        self.assertLess(one_loop_time, two_loop_time)
        self.assertLess(no_loop_time * 10, one_loop_time)
