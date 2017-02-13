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
        # TODO: it's unclear if the gradient calculation with the weight shape
        # needs to be done here.
        # dW = np.zeros(W.shape) # initialize the gradient as zero
        dW = 0

        # compute the loss and the gradient
        num_classes = batch_scores.shape[0]
        num_points = batch_scores.shape[1]
        loss = 0.0
        for i in range(num_points):
            correct_class_score = batch_scores[batch_labels[i]][i]
            for j in range(num_classes):
                if j == batch_labels[i]:
                    continue
                margin = batch_scores[j][i] - correct_class_score + 1    # note delta = 1
                if margin > 0:
                    loss += margin

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        loss /= num_points

        #############################################################################
        # TODO:                                                                     #
        # Compute the gradient of the loss function and store it dW.                #
        # Rather that first computing the loss and then computing the derivative,   #
        # it may be simpler to compute the derivative at the same time that the     #
        # loss is being computedd. As a result you may need to modify some of the   #
        # code above to compute the gradient.                                       #
        #############################################################################

        return loss, dW

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

        local_batch_scores = batch_scores
        num_points = batch_labels.shape[0]
        correct_classification_indices = (batch_labels, np.arange(num_points))
        correct_classifications = batch_scores[correct_classification_indices]
        local_batch_scores = batch_scores + 1 - correct_classifications
        local_batch_scores = np.maximum(local_batch_scores, 0)
        local_batch_scores[correct_classification_indices] = 0
        loss = np.sum(np.sum(local_batch_scores, axis=0)) / num_points

        self.gradient[local_batch_scores < 0] = 0
        self.gradient[local_batch_scores > 0] = 1
        self.gradient[correct_classification_indices] = \
            -1 * np.sum(self.gradient, axis=0)

        return loss, self.gradient
