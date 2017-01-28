import random
import numpy as np

import unittest
from samplers.sampler_random import SamplerRandom


class TestSamplerRandom(unittest.TestCase):

  def setUp(self):
    self.sampler = SamplerRandom()

  def test_sampler_random(self):
    trial_bounds = 100
    for trial in range(trial_bounds):
      self.sampler.seed(random.randint(0, 100))
      points_size = random.randint(1, trial_bounds)
      sample_size = random.randint(1, points_size)
      points = np.zeros(points_size)
      sample = self.sampler.sample(points, sample_size)

      self.assertEqual(points.shape[0], points_size)
      self.assertEqual(sample.shape[0], sample_size)
