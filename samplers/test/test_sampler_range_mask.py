import random
import numpy as np

import unittest
from samplers.sampler_range_mask import SamplerRangeMask


class TestSamplerRangeMask(unittest.TestCase):

  def setUp(self):
    self.sampler = SamplerRangeMask()

  def test_sampler_range_mask_dimensionality(self):
    """
    Test that the dimensionality (shape) of the data-points passed to
    SamplerRangeMask matches the dimensionality of its output, save
    the last (sampled) dimension.
    """

    # Here it becomes clear -- we need glue to feed data into the
    # sampler.  Testing the stub can proceed independently of the glue
    #
    # Presumably, `points` in `sample` would contain the output of
    # some importer (multi-dimensional data).  For the RangeMask
    # Sampler, the data is a one-dimensional array.

    trial_bounds = 100
    for trial in range(trial_bounds):
      points_size = random.randint(0,trial_bounds)
      sample_size = random.randint(0,points_size)
      points = np.zeros(points_size)
      sample = self.sampler.sample(points,sample_size)

      self.assertEqual(points.shape[:-1], sample.shape[:-1])
      self.assertLessEqual(sample.shape[-1], points.shape[-1])
