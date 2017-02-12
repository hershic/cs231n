class DatasetTrain:
    """
    A training dataset containing a train group and a test group.
    """
   def __init__(self, train_points, train_labels, test_points,
                test_labels):
        self.train_points = train_points
        self.train_labels = train_labels
        self.test_points = test_points
        self.test_labels = test_labels

    def train(self):
        return train

    def test(self):
        return test
