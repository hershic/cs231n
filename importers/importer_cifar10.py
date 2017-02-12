from datasets.dataset_simple import DatasetSimple
from importers.importer_base import ImporterBase
import six.moves.cPickle as pickle
import numpy as np
import os


class ImporterCIFAR10(ImporterBase):
  def __init__(self, dir):
    self.dir = dir

  def import_all(self):
    # FIXME: move to saner
    # points, labels = zip(*self.import_batch())
    points = []
    labels = []
    for batch in self.import_batch():
      points.append(batch[0])
      labels.append(batch[1])
    return DatasetSimple(np.concatenate(points), np.concatenate(labels))

  def import_batch(self):
    for batch in range(1, 6):
      filename = os.path.join(self.dir, 'data_batch_%d' % (batch, ))
      yield self.import_batch_file(filename)
    yield self.import_batch_file(os.path.join(self.dir, 'test_batch'))
    return

  def import_batch_file(self, filename):
    with open(filename, 'rb') as f:
      datadict = pickle.load(f, encoding='latin1')
      data = datadict['data']
      labels = datadict['labels']
      data = data \
        .reshape(-1, 3, 32, 32) \
        .transpose(0, 2, 3, 1) \
        .astype("float")
      labels = np.array(labels)
      return data, labels
