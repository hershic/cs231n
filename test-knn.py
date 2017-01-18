import unittest
import random
import numpy as np
from data_utils import load_CIFAR10
from classifiers import KNearestNeighbor

cifar10_dir = 'datasets/cifar-10-batches-py'

class TestSomething(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train, self.X_test, self.y_test = load_CIFAR10(cifar10_dir)
        self.accuracy = 0.0
        pass

    def subsample(self):
        num_training = 5000
        mask = range(num_training)
        X_train = self.X_train[mask]
        y_train = self.y_train[mask]

        num_test = 500
        mask = range(num_test)
        X_test = self.X_test[mask]
        y_test = self.y_test[mask]
        return (X_train, y_train, X_test, y_test, num_training, num_test)

    @unittest.skip("done with this guy")
    def test_dist_two_loops(self):
        (X_train, y_train, X_test, y_test, num_training, num_test) = self.subsample()

        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))

        classifier = KNearestNeighbor()
        classifier.train(X_train, y_train)
        dists = classifier.compute_distances_two_loops(X_test)

        # Now implement the function predict_labels and run the code below:
        # We use k = 1 (which is Nearest Neighbor).
        y_test_pred = classifier.predict_labels(dists, k=5)

        # Compute and print the fraction of correctly predicted examples

        num_correct = np.sum(y_test_pred == y_test)
        accuracy = float(num_correct) / num_test
        print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

    def test_dist_one_loop(self):
        (X_train, y_train, X_test, y_test, num_training, num_test) = self.subsample()
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))

        classifier = KNearestNeighbor()
        classifier.train(X_train, y_train)

        dists = classifier.compute_distances_two_loops(X_test)

        # Now lets speed up distance matrix computation by using partial vectorization
        # with one loop. Implement the function compute_distances_one_loop and run the
        # code below:
        dists_one = classifier.compute_distances_one_loop(X_test)

        # To ensure that our vectorized implementation is correct, we make sure that it
        # agrees with the naive implementation. There are many ways to decide whether
        # two matrices are similar; one of the simplest is the Frobenius norm. In case
        # you haven't seen it before, the Frobenius norm of two matrices is the square
        # root of the squared sum of differences of all elements; in other words, reshape
        # the matrices into vectors and compute the Euclidean distance between them.
        difference = np.linalg.norm(dists - dists_one, ord='fro')
        print 'Difference was: %f' % (difference, )
        if difference < 0.001:
          print 'Good! The distance matrices are the same'
        else:
          print 'Uh-oh! The distance matrices are different'


    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
