import unittest
from importers.cifar10 import ImporterCIFAR10


class TestImporterCIFAR10(unittest.TestCase):
  def test_import(self):
    importer = ImporterCIFAR10('datasets/cifar-10-batches-py/')
    batches = 0
    for batch in importer.import_batch():
      self.assertEqual(batch[0].shape, (10000, 32, 32, 3))
      self.assertEqual(batch[1].shape, (10000,))
      batches += 1
    self.assertEqual(batches, 6)
