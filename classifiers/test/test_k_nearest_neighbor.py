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

        x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)
        self.accuracy = 0.0
        (x_train, x_test, y_train, y_test) = \
            self.subsample(x_train, x_test, y_train, y_test, self.num_train, self.num_test)


        self.y_train = y_train
        self.y_test = y_test
        self.x_train = np.reshape(x_train, (x_train.shape[0], -1))
        self.x_test = np.reshape(x_test, (x_test.shape[0], -1))

        self.classifier = KNearestNeighbor()
        self.classifier.train(self.x_train, self.y_train)


    def subsample(self, x_train, x_test, y_train, y_test, num_train, num_test):
        mask = range(num_train)
        x_train = x_train[mask]
        y_train = y_train[mask]

        mask = range(num_test)
        x_test = x_test[mask]
        y_test = y_test[mask]

        return (x_train, x_test, y_train, y_test)


    def test_dist_two_loops(self):
        dists_two = self.classifier.compute_distances_two_loops(self.x_test)

        # Now implement the function predict_labels and run the code below:
        # We use k = 1 (which is Nearest Neighbor).
        y_test_pred = self.classifier.predict_labels(dists_two, k=5)

        # Compute and print the fraction of correctly predicted examples

        # accuracy should be somewhere around 15-25% depending on the
        # subsampling pattern
        num_correct = np.sum(y_test_pred == self.y_test)
        accuracy = float(num_correct) / self.num_test
        self.assertGreater(num_correct, 1)
        self.assertLess(accuracy, 0.3)
        self.assertGreater(accuracy, 0.1)


    def test_dist_one_loop(self):
        # Now lets speed up distance matrix computation by using partial vectorization
        # with one loop. Implement the function compute_distances_one_loop and run the
        # code below:
        dists_two = self.classifier.compute_distances_two_loops(self.x_test)
        dists_one = self.classifier.compute_distances_one_loop(self.x_test)

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
        dists_one = self.classifier.compute_distances_one_loop(self.x_test)
        dists_none = self.classifier.compute_distances_no_loops(self.x_test)

        # check that the distance matrix agrees with the one we computed before:
        difference = np.linalg.norm(dists_one - dists_none, ord='fro')
        self.assertAlmostEqual(difference, 0)
        print 'Difference was: %f' % (difference, )
        if difference < 0.001:
          print 'Good! The distance matrices are the same'
        else:
          print 'Uh-oh! The distance matrices are different'
