from math import sqrt

# The amount of data DATA_NOISE to add to the variance when using weighted voting
DATA_NOISE = 10 ** (-5)


class KNNClassifier:
    def __init__(self, space):
        """
        Initializes a KNNClassifier object with the given space.

        Args:
            space (Space): The space object containing the points to be classified.
        """
        self.space = space

    def sort_by_distance(self, p1):
        """
        Sorts the data points in the space by their Euclidean distance to the given point.

        Args:
            p1 (Point): The point to sort by distance to.
        """
        self.space.points.sort(key=lambda x: KNNClassifier.calc_distance(x, p1))

    def classify(self, p1, k):
        """
        Classifies the given point using the K-Nearest Neighbors algorithm.

        Args:
            p1 (Point): The point to be classified.
            k (int): The number of nearest neighbors to use for classification.

        Returns:
            str: The classification label for the given point.
        """
        self.sort_by_distance(p1)
        k_nearest_points = self.space.points[:k]

        votes = {}
        for point in k_nearest_points:
            distance = KNNClassifier.calc_distance(p1, point)

            # Determine number of votes based on distance and whether weighted voting is enabled
            points = max(1, 1 / (distance + DATA_NOISE))

            if point.classifier in votes:
                votes[point.classifier] += points

            else:
                votes[point.classifier] = points

        # Return the classification with the most votes
        return max(votes, key=votes.get)

    @staticmethod
    def calc_distance(p1, p2):
        """
        Calculates the Euclidean distance between two points.

        Args:
            p1 (Point): The first point.
            p2 (Point): The second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        # Define a function to calculate the distance between two coordinates
        calc = lambda x1, x2: pow(x1 - x2, 2)

        # Calculate the distance between each dimension of the two points and sum them
        return sqrt(sum(map(calc, p1.dimensions, p2.dimensions)))
