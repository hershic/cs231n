import numpy as np

from datasets.dataset_train import DatasetTrain


class PartitionerKFolds:

  def __init__(self):
    # Don't need to do anything here.
    pass

  def partition(self, data, num_folds, roll_amount):
    points_split = np.array_split(data.points, num_folds)
    labels_split = np.array_split(data.labels, num_folds)

    return DatasetTrain(points_split[0], labels_split[0],
                        np.concatenate(points_split[1:]),
                        np.concatenate(labels_split[1:]))
