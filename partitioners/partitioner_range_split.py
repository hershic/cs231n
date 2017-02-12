import numpy as np
import math

from datasets.dataset_train import DatasetTrain

class PartitionerRangeSplit:

  def __init__(self):
    # Don't need to do anything here.
    pass

  def partition(self, data, split_ratio):
    total_datum = data.points.shape[0]
    num_test = math.floor(total_datum * split_ratio)
    num_train = math.ceil(total_datum * (1 - split_ratio))

    return DatasetTrain(data.points[range(num_test, num_test + num_train)],
                        data.labels[range(num_test, num_test + num_train)],
                        data.points[range(num_test)],
                        data.labels[range(num_test)])
