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

      data_size = random.randint(2, trial_bounds)
      sample_size = random.randint(1, data_size - 1)
      data = {}
      data['points'] = np.zeros((data_size, 10))
      data['labels'] = np.zeros(data_size)
      sampled_data = self.sampler.sample(data, sample_size)

      self.assertIsNotNone(data['points'])
      self.assertIsNotNone(data['labels'])
      self.assertIsNotNone(sampled_data['points'])
      self.assertIsNotNone(sampled_data['labels'])

      self.assertGreater(data['points'].shape[0], sampled_data['points'].shape[0])
      self.assertGreater(data['labels'].shape[0], sampled_data['labels'].shape[0])

      self.assertEqual(data['points'].shape[0], data_size)
      self.assertEqual(data['labels'].shape[0], data_size)

      self.assertEqual(sampled_data['points'].shape[0], sample_size)
      self.assertEqual(sampled_data['labels'].shape[0], sample_size)
