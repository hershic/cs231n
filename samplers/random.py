import numpy as np
from samplers.base import SamplerBase


class SamplerRandom(SamplerBase):

    def __init__(self):
        pass

    def seed(self, seed_value):
        np.random.seed(seed_value)

    def sample(self, dataset, num):
        mask = np.random.choice(dataset['points'].shape[0], num, replace=False)
        return {
            'points': dataset['points'][mask],
            'labels': dataset['labels'][mask]
        }
