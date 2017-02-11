import numpy as np


class LayerFullyConnected:
    """
    Models a fully connected Neural Network layer where each input point is
    connected to each output score with some weight.

    Inputs:
    - num_inputs: The number of input points in each dataset point, i.e. the
      number of pixels in an input image. (point_size,)
    - num_outputs: The number of output activation points the network should
      return, i.e. the number of classifications available.
      (num_classifications,)
    """
    def __init__(self, num_inputs, num_outputs, weightsInit=None, gradientInit=None):
        if (weightsInit is not None):
            self.weights = weightsInit
        else:
            self.weights = np.random.randn(num_outputs, num_inputs) * 1e-4

        if (not gradientInit):
            self.gradient = np.random.randn(num_outputs, num_inputs) * 1e-4
        else:
            self.gradient = gradientInit

    def forward_naive(self, input_points):
        """
        Computes the forward pass of a fully-connected layer with the input
        weights in a naive-unvectorized implementation, returning the category
        scores for the input points.

        Inputs:
        - points: (num_points, point_size)
        Outputs:
        - scores: (num_classifications, num_points)
        """
        num_outputs = self.weights.shape[0]
        num_inputs = input_points.shape[0]
        scores = np.zeros((num_inputs, num_outputs))
        for i in range(num_inputs):
            scores[i] = self.weights.dot(input_points[i])
        return scores.T

    def forward_vectorized(self, input_points):
        """
        Computes the forward pass of a fully-connected layer with the input
        weights in a vectorized implementation, returning the category scores
        for the input points.

        Inputs:
        - points: (num_points, point_size)
        Outputs:
        - scores: (num_classifications, num_points)
        """
        # TODO obtain and cache the gradient
        return self.weights.dot(input_points.T)
