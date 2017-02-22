import numpy as np


class RegularizerL2():
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def calculate(self):
        regularization = 0.0
        for layer in self.layers:
            regularization += 0.5 * np.sum(layer.weights * layer.weights)
        return regularization
