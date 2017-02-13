import numpy as np


class PreprocessorCumulativeMovingAverage:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.batch_counter = 0

    def calculate_batch(self, batch_points):
        self.mean = (np.mean(batch_points, axis=0) + self.batch_counter * self.mean) \
            / (self.batch_counter + 1)
        self.batch_counter += 1

    def normalize_batch(self, batch_points):
        batch_points -= self.mean
