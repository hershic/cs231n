import unittest
import random
import numpy as np
from util.data import load_CIFAR10
from classifiers import KNearestNeighbor

cifar10_dir = 'datasets/cifar-10-batches-py'

class TestSomething(unittest.TestCase):
  def setUp(self):
    self.num_train = 500
    self.num_test = 50

    train_points, train_labels, test_points, test_labels = load_CIFAR10(cifar10_dir)
    self.accuracy = 0.0
    (train_points, train_labels, test_points, test_labels) = \
      self.subsample(train_points, train_labels, test_points, test_labels, self.num_train, self.num_test)

    self.train_points = np.reshape(train_points, (train_points.shape[0], -1))
    self.train_labels = train_labels
    self.test_points = np.reshape(test_points, (test_points.shape[0], -1))
    self.test_labels = test_labels

    self.classifier = KNearestNeighbor()
    self.classifier.train(self.train_points, self.train_labels)


  def subsample(self, train_points, train_labels, test_points, test_labels, num_train, num_test):
    mask = range(num_train)
    train_points = train_points[mask]
    train_labels = train_labels[mask]

    mask = range(num_test)
    test_points = test_points[mask]
    test_labels = test_labels[mask]

    return (train_points, train_labels, test_points, test_labels)


  def test_dist_two_loops(self):
    dists_two = self.classifier.compute_distances_two_loops(self.test_points)

    # Now implement the function predict_labels and run the code below:
    # We use k = 1 (which is Nearest Neighbor).
    test_labels_pred = self.classifier.predict_labels(dists_two, k=5)

    # Compute and print the fraction of correctly predicted examples

    # accuracy should be somewhere around 15-25% depending on the
    # subsampling pattern
    num_correct = np.sum(test_labels_pred == self.test_labels)
    accuracy = float(num_correct) / self.num_test
    self.assertGreater(num_correct, 1)
    self.assertLess(accuracy, 0.3)
    self.assertGreater(accuracy, 0.1)


  def test_dist_one_loop(self):
    # Now lets speed up distance matrix computation by using partial vectorization
    # with one loop. Implement the function compute_distances_one_loop and run the
    # code below:
    dists_two = self.classifier.compute_distances_two_loops(self.test_points)
    dists_one = self.classifier.compute_distances_one_loop(self.test_points)

    # To ensure that our vectorized implementation is correct, we make sure that it
    # agrees with the naive implementation. There are many ways to decide whether
    # two matrices are similar; one of the simplest is the Frobenius norm. In case
    # you haven't seen it before, the Frobenius norm of two matrices is the square
    # root of the squared sum of differences of all elements; in other words, reshape
    # the matrices into vectors and compute the Euclidean distance between them.
    difference = np.linalg.norm(dists_two - dists_one, ord='fro')
    self.assertAlmostEqual(difference, 0.0)


  def test_dist_no_loops(self):
    # Now implement the fully vectorized version inside compute_distances_no_loops
    # and run the code
    dists_one = self.classifier.compute_distances_one_loop(self.test_points)
    dists_none = self.classifier.compute_distances_no_loops(self.test_points)

    # check that the distance matrix agrees with the one we computed before:
    difference = np.linalg.norm(dists_one - dists_none, ord='fro')
    self.assertAlmostEqual(difference, 0.0)

