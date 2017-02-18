import numpy as np


class LayerFullyConnected:
    """
    Models a fully connected Neural Network layer where each input point is
    connected to each output score with some weight.

    Inputs:
    - input_dim: The number of input points in each dataset point, e.g. the
        number of pixels in an input image. (points_per_datum,)
    - output_dim: The number of output activation points the network should
        return, e.g. the number of classifications available.
        (num_classifications,)
    """
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 1e-4
        self.bias = np.random.randn(output_dim) * 1e-4
        self.gradient = np.zeros((output_dim, input_dim))

    def forward_naive(self, batch_points):
        """
        Computes the forward pass of a fully-connected layer with the input
        weights in a vectorized implementation, returning the category scores
        for the input points. The input points is in the shape (batch_size,
        points_per_datum). batch_size refers to, for example, the number of
        images in the input. points_per_datum refers to, for example, the size
        of each input image.

        Inputs:
        - batch_points: (batch_size, points_per_datum)
        Outputs:
        - scores: (num_classifications, batch_size)
        """
        batch_size = batch_points.shape[0]
        num_outputs = self.weights.shape[1]
        scores = np.zeros((batch_size, num_outputs))
        for i in range(batch_size):
            scores[i] = batch_points[i].dot(self.weights)
        return scores + self.bias

    def forward_vectorized(self, batch_points):
        """
        Computes the forward pass of a fully-connected layer with the input
        weights in a vectorized implementation, returning the category scores
        for the input points. The input points is in the shape (batch_size,
        points_per_datum). batch_size refers to, for example, the number of
        images in the input. points_per_datum refers to, for example, the size
        of each input image.

        Inputs:
        - points: (batch_size, points_per_datum)
        Outputs:
        - scores: (num_classifications, batch_size)
        """
        # TODO obtain and cache the gradient
        self.gradient = self.weights
        return batch_points.dot(self.weights) + self.bias
