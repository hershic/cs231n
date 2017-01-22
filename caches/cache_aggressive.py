from collections import deque
import numpy as np


class CacheAggressive:
  """
  CacheAggressive loads all the data at once and gives you chunks as you
  request for them.
  """
  def __init__(self, importer):
    self.importer = importer
    self.data = deque()
    for batch in importer.shape[0]:
      pass

  def fetch(self, amount):
    buffer = deque()
    start_pointer = 0

    for batch in self.importer.import_batch():
      buffer.extend(batch)
      if (len(buffer) >= start_pointer * amount + amount):
        yield buffer[range(start_pointer, start_pointer + amount)]
        start_pointer += 1

    yield None
