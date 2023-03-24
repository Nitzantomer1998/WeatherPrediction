# Library to calculate square root
from math import sqrt

# The amount of data_noise to add to the distance when using weighted voting
data_noise = 10 ** (-5)


class KNNClassifier:
    def __init__(self, space):
        """
        Initializes a KNNClassifier object with the given space.

        Args:
            space (Space): The space object containing the points to be classified.
        """
        self.space = space

    def classify(self, query_point, k: int, weighted: bool) -> str:
        """
        Classifies a point using the k-nearest neighbors algorithm.

        Args:
            query_point (Point): The point to be classified.
            k (int): The number of nearest neighbors to consider.
            weighted (bool): Whether to use distance-weighted voting.

        Returns:
            str: The classification label for the given point.
        """
        # Sort the points by distance from the query point
        self.sort_points_by_distance(query_point)
        k_nearest_points = self.space.points[:k]

        # Count the votes for each class based on the k-nearest points
        votes = {}
        for point in k_nearest_points:
            distance = self.calculate_distance(query_point, point)
            weight = (1 if not weighted else max(1, 1 / (distance + data_noise)))

            if point.classifier in votes:
                votes[point.classifier] += weight
            else:
                votes[point.classifier] = weight

        # Return the class with the most votes
        return max(votes, key=votes.get)

    def sort_points_by_distance(self, query_point):
        """
        Sorts the points in the space by their distance from the given query point.

        Args:
            query_point (Point): The point to use as a reference for distance calculation.
        """
        self.space.points.sort(key=lambda p: self.calculate_distance(query_point, p))

    @staticmethod
    def calculate_distance(point1, point2) -> float:
        """
        Calculates the Euclidean distance between two points.

        Args:
            point1 (Point): The first point.
            point2 (Point): The second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        calc = lambda x1, x2: pow(x1 - x2, 2)
        return sqrt(sum(map(calc, point1.dimensions, point2.dimensions)))
