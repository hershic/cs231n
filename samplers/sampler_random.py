import numpy as np
from samplers.sampler_base import SamplerBase
from datasets.dataset_simple import DatasetSimple

class SamplerRandom(SamplerBase):

  def __init__(self):
    pass

  def seed(self, seed_value):
    np.random.seed(seed_value)

  def sample(self, dataset, num):
    mask = np.random.choice(dataset.points.shape[0], num, replace=False)
    return DatasetSimple(dataset.points[mask], dataset.labels[mask])
