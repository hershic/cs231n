import numpy as np
from random import shuffle
from classifiers.linear_classifier import LinearClassifier


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def train(self, points, labels):
        pass

    def loss(self, X_batch, y_batch, reg):
        return self.svm_loss_vectorized(self.W, X_batch, y_batch, reg)

    def svm_loss_naive(self, scores, labels, reg):
        """
        Structured SVM loss function, naive implementation (with loops).

        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X: A numpy array of shape (N, D) containing a minibatch of data.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means
            that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength

        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
        """
        # TODO: it's unclear if the gradient calculation with the weight shape
        # needs to be done here.
        # dW = np.zeros(W.shape) # initialize the gradient as zero
        dW = 0

        # compute the loss and the gradient
        num_classes = scores.shape[0]
        num_points = scores.shape[1]
        loss = 0.0
        for i in range(num_points):
            correct_class_score = scores[labels[i]][i]
            for j in range(num_classes):
                if j == labels[i]:
                    continue
                margin = scores[j][i] - correct_class_score + 1    # note delta = 1
                if margin > 0:
                    loss += margin

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        loss /= num_points

        # Add regularization to the loss.
        # TODO: understand what this is
        # loss += 0.5 * reg * np.sum(W * W)

        #############################################################################
        # TODO:                                                                     #
        # Compute the gradient of the loss function and store it dW.                #
        # Rather that first computing the loss and then computing the derivative,   #
        # it may be simpler to compute the derivative at the same time that the     #
        # loss is being computedd. As a result you may need to modify some of the   #
        # code above to compute the gradient.                                       #
        #############################################################################

        return loss, dW

    def svm_loss_vectorized(self, scores, labels, reg=1e-6):
        """
        Structured SVM loss function, vectorized implementation.

        Inputs and outputs are the same as svm_loss_naive.
        """

        # TODO: it's unclear if the gradient calculation with the weight shape
        # needs to be done here.
        # dWeights = np.zeros(weights.shape)    # initialize the gradient as zero
        dWeights = 0

        num_points = labels.shape[0]
        correct_classification_indices = (labels, np.arange(num_points))
        correct_classifications = scores[correct_classification_indices]
        scores += 1 - correct_classifications
        scores = np.maximum(scores, 0)
        scores[correct_classification_indices] = 0
        loss = np.sum(np.sum(scores, axis=0)) / num_points

        #############################################################################
        # TODO:                                                                     #
        # Implement a vectorized version of the gradient for the structured SVM     #
        # loss, storing the result in dW.                                           #
        #                                                                           #
        # Hint: Instead of computing the gradient from scratch, it may be easier    #
        # to reuse some of the intermediate values that you used to compute the     #
        # loss.                                                                     #
        #############################################################################
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return loss, dWeights

    def loss_vectorized(self, weights, points, labels):
        scores = np.transpose(weights).dot(points)
        margins = np.maximum(0, scores - scores[labels] + 1)
        margins[labels] = 0
        return np.sum(margins)

    def numerical_gradient_estimate(self, W, points, labels):
        # Once you've implemented the gradient, recompute it with the code below
        # and gradient check it with the function we provided for you

        # Compute the loss and its gradient at W.
        loss, grad = self.svm_loss_naive(W, points, labels, 0.0)

        # Numerically compute the gradient along several randomly chosen dimensions, and
        # compare them with your analytically computed gradient. The numbers should match
        # almost exactly along all dimensions.
        from lib.gradient_check import grad_check_sparse
        f = lambda w: self.svm_loss_naive(w, points, labels, 0.0)[0]
        grad_numerical = grad_check_sparse(f, W, grad)

        # do the gradient check once again with regularization turned on
        # you didn't forget the regularization gradient did you?
        loss, grad = self.svm_loss_naive(W, points, labels, 1e2)
        f = lambda w: self.svm_loss_naive(w, points, labels, 1e2)[0]
        grad_numerical = grad_check_sparse(f, W, grad)

        print("hello world")
