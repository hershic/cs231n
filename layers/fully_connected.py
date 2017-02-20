import numpy as np


class LayerFullyConnected:
    """
    Models a fully connected Neural Network layer where each input point is
    connected to each output score with some weight.

    Inputs:
    - transformation_shape: The shape of the transformation between input
      points, e.g. the number of pixels in an input image and output points,
      e.g. the number of classifications available.
    """
    def __init__(self, transformation_shape):
        self.weights = np.random.randn(*transformation_shape) * 1e-2
        # "He" initialization:
        # https://arxiv.org/abs/1502.01852
        self.bias = np.random.randn(transformation_shape[1]) \
                    * np.sqrt(2.0 / transformation_shape[1])

        self.d_weights = np.ones(transformation_shape)
        self.d_bias = np.ones(transformation_shape[1])

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

    def forward(self, batch_points):
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

    def backward(self, gradient):
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
