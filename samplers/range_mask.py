from samplers.base import SamplerBase


class SamplerRangeMask(SamplerBase):

    def __init__(self):
        pass

    def seed(self, seed_value):
        # The range mask sampler doesn't do anything with the seed.
        pass

    def sample(self, dataset, num, start=0):
        return {
            'points': dataset['points'][start:start + num],
            'labels': dataset['labels'][start:start + num]
        }
