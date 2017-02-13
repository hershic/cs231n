import numpy as np


class LayerFullyConnected:
    """
    Models a fully connected Neural Network layer where each input point is
    connected to each output score with some weight.

    Inputs:
    - num_inputs: The number of input points in each dataset point, i.e. the
        number of pixels in an input image. (points_per_datum,)
    - num_outputs: The number of output activation points the network should
        return, i.e. the number of classifications available.
        (num_classifications,)
    """
    def __init__(self, num_inputs, num_outputs):
        self.weights = np.random.randn(num_outputs, num_inputs) * 1e-4
        self.gradient = np.zeros((num_outputs, num_inputs))

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
        num_outputs = self.weights.shape[0]
        num_inputs = batch_points.shape[0]
        scores = np.zeros((num_inputs, num_outputs))
        for i in range(num_inputs):
            scores[i] = self.weights.dot(batch_points[i])
        return scores.T

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
        return self.weights.dot(batch_points.T)
