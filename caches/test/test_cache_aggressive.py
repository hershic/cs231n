import unittest
from caches.cache_aggressive import CacheAggressive
from importers.importer_dummy import ImporterDummy


class TestCacheAggressive(unittest.TestCase):
  def __init__(self):
    self.cache = CacheAggressive(ImporterDummy('some-file'))

  def test_caching(self):
    pass
