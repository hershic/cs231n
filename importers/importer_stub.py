import numpy as np
from itertools import chain
from importers.importer_base import ImporterBase


class ImporterStub(ImporterBase):
  # Data shape is in the shape of (NUM, DATA_SHAPE), due to NUM elements of
  # DATA_SHAPE shape; classification shape is in the size of NUM
  DATA_SHAPE = (3, 5, 5)
  CLASS_SHAPE = 10

  # abc
  def __init__(self, filename):
    self.filename = filename
    # do a filesystem scan to find out how many batches there are

  # abc
  def import_batch(self):
    for num in range(10):
      size = [(self.CLASS_SHAPE,), self.DATA_SHAPE]
      yield (np.zeros(self.CLASS_SHAPE),
             np.zeros(list(chain(*size))))
