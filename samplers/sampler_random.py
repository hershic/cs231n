import numpy as np
from samplers.sampler_base import SamplerBase


class SamplerRandom(SamplerBase):

  def __init__(self):
    pass

  def seed(self, seed_value):
    np.random.seed(seed_value)

  def sample(self, points, num):
    mask = np.random.choice(points.shape[0], num, replace=False)
    return points[mask]
