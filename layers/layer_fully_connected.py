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

        self.d_weights = np.ones((output_dim, input_dim))
        self.d_bias = np.ones((output_dim,))

    def _cache_gradients(self, batch_points):
        """
        Caches the partial gradients of the batch_points, weights, and bias
        with respect to the outputs.
        """
        self.d_batch_points = self.weights
        self.d_weights = batch_points
        self.d_bias = np.ones(self.d_bias.shape)

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

        Side-Effects:
        - Computes and stores the partial gradient of the output with respect
          to the weights, bias, and inputs.
        """
        self._cache_gradients(batch_points)

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

        Side-Effects:
        - Computes and stores the partial gradient of the output with respect
          to the weights, bias, and inputs.
        """
        self._cache_gradients(batch_points)
        return batch_points.dot(self.weights) + self.bias

    def backward_vectorized(self, gradient):
        """
        Computes the backward pass of a fully-connected layer with the partial
        gradient from the subsequent layer and returning the partial gradient
        with respect to the previous layer's inputs.

        Inputs:
        - gradient: (batch_size, points_per_datum)

        Outputs:
        - d_batch_points: (input_dim, output_dim)

        Side-Effects:
        - Computes and stores the partial gradient of the complete output with
          respect to the weights, bias, and inputs by using the chain rule
          against the input gradient from the subsequent layer (and its
          subsequent layers).
        """
        self.d_batch_points = gradient.dot(self.d_batch_points.T)
        self.d_weights = self.d_weights.T.dot(gradient)
        self.d_bias = np.sum(self.d_bias * gradient, axis=0)
        return self.d_batch_points
