import numpy as np
from random import shuffle


class ClassifierSoftmax():
    def __init__(self, scores_shape):
        self.gradient = np.zeros(scores_shape)

    def softmax_loss_naive(self, scores, labels):
        """
        Softmax loss function, naive implementation (with loops)

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
        # Initialize the loss and gradient to zero.
        loss = 0.0

        # If we interpret the input data as unnormalized log probablities, then
        # we must exponentiate the probabilities to obtain the unnormalized
        # probabilities, then normalize them to get the probabilities of a
        # particular image belonging to a certain class. The softmax function
        # therefore looks like:

        # $$L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)$$

        num_classes = scores.shape[0]
        num_points = scores.shape[1]

        # First we exponentiate each parameter.
        local_scores = np.exp(scores)

        # Second we compure the scores sum (for later use).
        scores_sum = np.sum(local_scores, axis=0)

        # Third, using the scores sums, compute the gradient of the softmax
        # function. The gradient looks like:

        # $$\frac{\text{d}L_i}{\text{d}s_j} = - \frac{\sum_{k, k \ne y_i} e^{s_k}}{\sum_k e^{s_k}}$$
        # if $j = y_i$
        # $$ \frac{\text{d}L_i}{\text{d}s_j} = - \frac{e^{s_j}}{\sum_k e^{s_k}} $$
        # if $j \ne y_i$

        # e.g., if $y_i = 2$, then
        # $$L_i = -\log \left(\frac{e^{s_2}}{e^{s_0} + e^{s_0} + e^{s_2}} \right)$$
        # so
        # $$\frac{\text{d}L_i}{\text{d}s_0} = \frac{e^{s_0}}{\sum_k e^{s_k}}$$
        # and
        # $$\frac{\text{d}L_i}{\text{d}s_2} = -\frac{e^{s_0} + e^{s_1}}{\sum_k e^{s_k}}$$

        # The naive calculation is below. We store the gradient in self.
        for i in range(num_classes):
            for j in range(num_points):
                if i == labels[j]:
                    # this is really bad because subtraction allows for
                    # numerical instability and dropoff, but it's good enough
                    # for a naive approach
                    self.gradient[i, j] = -1 * (scores_sum[j] - local_scores[i, j]) / scores_sum[j]
                else:
                    self.gradient[i, j] = local_scores[i, j] / scores_sum[j]

        # Third we normalize each parameter with respect to the other
        # parameters.
        for i in range(num_classes):
            for j in range(num_points):
                local_scores[i, j] /= scores_sum[j]

        # Finally, we obtain our final loss.
        correct_classification_indices = (labels, np.arange(num_points))
        loss = np.sum(-1 * np.log(local_scores[correct_classification_indices]))

        return loss, self.gradient

    def softmax_loss_vectorized(self, X, y):
        """
        Softmax loss function, vectorized version.

        Inputs and outputs are the same as softmax_loss_naive.
        """
        # Initialize the loss and gradient to zero.
        loss = 0.0
        # dW = np.zeros_like(W)
        dW = 0.0

        #############################################################################
        # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
        # Store the loss in loss and the gradient in dW. If you are not careful     #
        # here, it is easy to run into numeric instability. Don't forget the        #
        # regularization!                                                           #
        #############################################################################
        pass
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW
