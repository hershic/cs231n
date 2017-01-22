import unittest
from importers.importer_stub import ImporterStub


class TestImporterStub(unittest.TestCase):
  def test_import(self):
    importer = ImporterStub('some-filename')
    batches = 0
    for batch in importer.import_batch():
      batches += 1
      self.assertEqual(batch[0].shape, (10,))
      self.assertEqual(batch[1].shape, (10, 3, 5, 5))
    self.assertGreater(batches, 0)
