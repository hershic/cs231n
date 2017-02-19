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
        - scores: A numpy array of shape (N, D) containing the scores of a batch of data.
        - labels: A numpy array of shape (N,) containing training labels; labels[i] = c
            means that scores[i] has label c, where 0 <= c < number of labels in scores.

        Returns a tuple of:
        - loss as single float
        - gradient with respect to the scores, same shape as the scores (N, D)
        """
        # Initialize the loss and gradient to zero.
        loss = 0.0

        # If we interpret the input data as unnormalized log probablities, then
        # we must exponentiate the probabilities to obtain the unnormalized
        # probabilities, then normalize them to get the probabilities of a
        # particular image belonging to a certain class. The softmax function
        # therefore looks like:

        # $$L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)$$

        num_points = scores.shape[0]
        num_classes = scores.shape[1]

        # First we exponentiate each parameter.
        local_scores = np.exp(scores)

        # Second we compute the scores sum (for later use).
        scores_sum = np.sum(local_scores, axis=1)

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
        for i in range(num_points):
            for j in range(num_classes):
                if j == labels[i]:
                    # this is really bad because subtraction allows for
                    # numerical instability and dropoff, but it's good enough
                    # for a naive approach
                    self.gradient[i, j] = -1 * (scores_sum[i] - local_scores[i, j]) / scores_sum[i]
                else:
                    self.gradient[i, j] = local_scores[i, j] / scores_sum[i]

        # Third we normalize each parameter with respect to the other
        # parameters.
        for i in range(num_points):
            for j in range(num_classes):
                local_scores[i, j] /= scores_sum[i]

        # Finally, we obtain our final loss.
        correct_classification_indices = (np.arange(num_points), labels)
        loss = np.sum(-1 * np.log(local_scores[correct_classification_indices])) / num_points

        return loss, self.gradient

    def softmax_loss_vectorized(self, scores, labels):
        """
        Softmax loss function, vectorized version.

        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - scores: A numpy array of shape (N, D) containing the scores of a batch of data.
        - labels: A numpy array of shape (N,) containing training labels; labels[i] = c
            means that scores[i] has label c, where 0 <= c < number of labels in scores.

        Returns a tuple of:
        - loss as single float
        - gradient with respect to the scores, same shape as the scores (N, D)
        """
        # Initialize the loss to zero.
        loss = 0.0
        num_points = scores.shape[0]

        # Before we do anything: Dividing large numbers is potentially
        # numerically unstable. Normalize the numbers by subtracting the
        # maximum from the scores.
        # ref: http://cs231n.github.io/linear-classify/#softmax
        local_scores = scores - np.max(scores)

        # If we interpret the input data as unnormalized log probablities, then
        # we must exponentiate the probabilities to obtain the unnormalized
        # probabilities, then normalize them to get the probabilities of a
        # particular image belonging to a certain class. The softmax function
        # therefore looks like:

        # $$L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)$$

        # First we exponentiate each parameter.
        local_scores = np.exp(local_scores)

        # Second we compute the scores sum (for later use).
        scores_sum = np.sum(local_scores, axis=1)

        # Third normalize the scores. Both the gradient general case ($j \ne
        # y_i$) and the loss depends on the normalized scores.
        correct_classification_indices = (np.arange(num_points), labels)
        normalized_scores = (local_scores.T / scores_sum).T
        loss = np.sum(-1 * np.log(normalized_scores[correct_classification_indices])) / num_points

        # Fourth compute the gradient. The gradient looks like:
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

        # The gradient general case ($j \ne y_i$) is simply the normalized
        # scores:
        self.gradient = normalized_scores

        # The gradient for the case when $j = y_i$ is a bit more complicated.
        # We need to sum all of the $j \ne y_i$ elements on the numerator, so
        # we generate a matrix which masks off just $j = y_i$. Then we do a
        # matrix-multiplication of that matrix and the exponentiated scores
        # matrix, sum the resultant matrix column-wise, and then divide that
        # vector by the scores_sum.
        noncorrect_classification_index_mask = np.ones(local_scores.shape)
        noncorrect_classification_index_mask[correct_classification_indices] = 0
        self.gradient[correct_classification_indices] = \
            -1 * np.sum(noncorrect_classification_index_mask * local_scores) / scores_sum

        # All done!
        return loss, self.gradient
