from samplers.sampler_base import SamplerBase


class SamplerRangeMask(SamplerBase):

  def __init__(self):
    pass

  def seed(self, seed_value):
    # The range mask sampler doesn't do anything with the seed.
    pass

  def sample(self, points, num, start=0):
    return points[start:start+num]
