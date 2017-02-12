from datasets.dataset_simple import DatasetSimple

class DatasetTrain:
    """
    A training dataset containing a train group and a test group.
    """
    def __init__(self, train_points, train_labels, test_points, test_labels):
        self.train = DatasetSimple(train_points, train_labels)
        self.test = DatasetSimple(test_points, test_labels)
