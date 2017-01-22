import numpy as np


class PartitionerKFolds:

  def __init__(self):
    # Don't need to do anything here.
    pass

  def partition(self, num_folds, points, labels, roll_amount):
    points_split = np.roll(np.array_split(points), roll_amount, axis=0)
    labels_split = np.roll(np.array_split(labels), roll_amount, axis=0)

    return {
      'train': {
        'points': points_split[0],
        'labels': labels_split[0]
      },
      'test': {
        'points': points_split[1:],
        'labels': labels_split[1:]
      }
    }
