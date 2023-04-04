class Space:
    def __init__(self) -> None:
        """
        Initializes an empty Space object.

        Attributes:
            points: A list to store the points in the space.
            classifications: A set to store the unique classifications of the points in the space.
        """
        self.points = []
        self.classifications = set()

    def add_point(self, point) -> None:
        """
        Adds a point to the space and updates the set of classifications.

        Args:
            point (Point): The point to be added to the space.
        """
        self.points.append(point)
        self.classifications.add(point.classifier)

    def clear_space(self) -> None:
        """
        Clears all points and classifications from the space.

        """
        self.points.clear()
        self.classifications.clear()
