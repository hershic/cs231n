import numpy as np


class LayerReLU:
    """
    Computes the activations of an input using Rectified Linear Units.
    """
    def __init__(self):
        pass

    def forward(self, input):
        """
        Computes the activations of an input using Rectified Linear Units.

        Input:
        - input: The numpy array to calculate activations from.

        Outputs:
        - activations: The ReLU activations from the inputs.

        Side-Effects:
        - Computes and stores the partial gradient of the output activations with respect
          to the input.
        """
        self.d_input = (np.sign(input) + 1) / 2
        return np.maximum(input, 0)

    def backward(self, gradient):
        return gradient * self.d_input
