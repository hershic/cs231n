import random
import numpy as np

import unittest
from samplers.range_mask import SamplerRangeMask


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
        # sampler.    Testing the stub can proceed independently of the glue
        #
        # Presumably, `points` in `sample` would contain the output of
        # some importer (multi-dimensional data).    For the RangeMask
        # Sampler, the data is a one-dimensional array.

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
