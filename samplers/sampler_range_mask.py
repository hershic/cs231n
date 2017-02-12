from samplers.sampler_base import SamplerBase

from datasets.dataset_simple import DatasetSimple


class SamplerRangeMask(SamplerBase):

  def __init__(self):
    pass

  def seed(self, seed_value):
    # The range mask sampler doesn't do anything with the seed.
    pass

  def sample(self, dataset, num, start=0):
    return DatasetSimple(dataset.points[start:start + num],
                         dataset.points[start:start + num])
