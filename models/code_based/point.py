class Point:
    def __init__(self, dimensions: tuple, classifier: str) -> None:
        """
        Initializes a Point object with given dimensions and a classifier.

        Args:
            dimensions (tuple): The tuple of values representing the point's position in n-dimensional space.
            classifier (str): The classification label of the point.
        """
        self.dimensions = dimensions
        self.classifier = classifier
