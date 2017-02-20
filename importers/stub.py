import numpy as np
from itertools import chain
from importers.base import ImporterBase


class ImporterStub(ImporterBase):
    # Data shape is in the shape of (num, data_shape), due to num elements of
    # data_shape shape; classification shape is in the size of num
    class_shape = 10
    data_shape = (3, 5, 5)

    def __init__(self, filename):
        self.filename = filename
        # do a filesystem scan to find out how many batches there are

    # generator
    def import_batch(self):
        for num in range(10):
            size = [(self.class_shape,), self.data_shape]
            yield (np.zeros(self.class_shape),
                         np.zeros(list(chain(*size))))
