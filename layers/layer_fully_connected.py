import numpy as np


class LayerFullyConnected:
    """
    Models a fully connected Neural Network layer where each input point is
    connected to each output score with some weight.
    """
    def __init__(self, weights_shape, points_shape, gradientInit=None):
        num_classifications = weights_shape[0]
        num_points = points_shape[0]

        # sane defaults
        if (not gradientInit):
            self.gradient = np.random.randn(num_classifications, num_points) * 1e-4
        else:
            self.gradient = gradientInit

    def forward_naive(self, weights, points):
        """
        Computes the forward pass of a fully connected layer with the input
        weights in a naive-unvectorized implementation, returning the category
        scores for the input points.

        Inputs:
        - weights: (num_classifications, point_size)
        - points: (num_points, point_size)
        Outputs:
        - scores: (num_points, num_classifications)
        """
        num_classifications = weights.shape[0]
        num_points = points.shape[0]
        scores = np.zeros((num_points, num_classifications))
        for i in range(num_points):
            scores[i] = weights.dot(points[i])
        return scores

    def forward_vectorized(self, weights, points):
        """
        Computes the forward pass of a fully connected layer with the input
        weights in a vectorized implementation, returning the category scores
        for the input points.

        Inputs:
        - weights: (num_classifications, point_size)
        - points: (num_points, point_size)
        Outputs:
        - scores: (num_points, num_classifications)
        """
        scores = weights.dot(points.T).T
        return scores
