import numpy as np
from random import shuffle
from classifiers.linear_classifier import LinearClassifier


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def __init__(self, batch_scores_shape):
        self.gradient = np.zeros(batch_scores_shape)

    def svm_loss_naive(self, batch_scores, batch_labels):
        """
        Structured SVM loss function, naive implementation (with loops).

        Inputs:
        - batch_scores: A numpy array of shape (N, D) containing the scores of
          a batch of data.
        - batch_labels: A numpy array of shape (N,) containing training
            batch_labels; batch_labels[i] = c means that batch_scores[i] has
            label c, where 0 <= c < number of batch_labels in batch_scores.

        Returns a tuple of:
        - loss as single float
        - gradient with respect to batch_scores in the same shape as
            batch_scores
        """

        # compute the loss and the gradient
        num_points = batch_scores.shape[0]
        num_classes = batch_scores.shape[1]
        loss = 0.0
        for i in range(num_points):
            correct_class_score = batch_scores[i, batch_labels[i]]
            for j in range(num_classes):
                if j == batch_labels[i]:
                    self.gradient[i, j] = -1 * num_classes + 1
                    continue
                # note delta = 1
                margin = batch_scores[i][j] - correct_class_score + 1
                if margin > 0:
                    loss += margin
                    self.gradient[i, j] = 1
                else:
                    self.gradient[i, j] = 0

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        loss /= num_points

        return loss, self.gradient

    def svm_loss_vectorized(self, batch_scores, batch_labels):
        """
        Structured SVM loss function, vectorized implementation.

        Inputs:
        - batch_scores: A numpy array of shape (N, D) containing the scores of
          a batch of data.
        - batch_labels: A numpy array of shape (N,) containing training
            batch_labels; batch_labels[i] = c means that batch_scores[i] has
            label c, where 0 <= c < number of batch_labels in batch_scores.

        Returns a tuple of:
        - loss as single float
        - gradient with respect to batch_scores in the same shape as
          batch_scores
        """

        num_points = batch_labels.shape[0]
        correct_classification_indices = (np.arange(num_points), batch_labels)
        correct_classifications = batch_scores[correct_classification_indices]
        local_batch_scores = (batch_scores.T + 1 - correct_classifications).T
        local_batch_scores = np.maximum(local_batch_scores, 0)
        local_batch_scores[correct_classification_indices] = 0
        loss = np.sum(np.sum(local_batch_scores, axis=1)) / num_points

        self.gradient[local_batch_scores < 0] = 0
        self.gradient[local_batch_scores > 0] = 1
        self.gradient[correct_classification_indices] = \
            -1 * np.sum(self.gradient, axis=1)

        return loss, self.gradient
