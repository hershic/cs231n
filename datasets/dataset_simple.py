class DatasetSimple:
    """
    A simple dataset consisting of points and labels.
    """
    def __init__(self, points, labels):
        self.points = points
        self.labels = labels

    def __str__(self):
        return "points: {0}\nlabels: {1}".format(self.points,
                                                 self.labels)
