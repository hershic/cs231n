import numpy as np


class PreprocessorCumulativeMovingAverage:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.batch_counter = 0

    def calculate_batch(self, points):
        self.mean = (np.mean(points, axis=0) + self.batch_counter * self.mean) \
            / (self.batch_counter + 1)
        self.batch_counter += 1

    def normalize_batch(self, points):
        points -= self.mean
