from math import sqrt

DATA_NOISE = 10 ** (-5)


class KNNClassifier:
    def __init__(self, space):
        self.space = space

    def sort_by_distance(self, p1):
        self.space.points.sort(key=lambda x: KNNClassifier.calc_distance(x, p1))

    def classify(self, p1, k):
        self.sort_by_distance(p1)
        k_nearest_points = self.space.points[:k]

        votes = {}
        for point in k_nearest_points:
            distance = KNNClassifier.calc_distance(p1, point)

            points = max(1, 1 / (distance + DATA_NOISE))

            if point.classifier in votes:
                votes[point.classifier] += points

            else:
                votes[point.classifier] = points

        return max(votes, key=votes.get)

    @staticmethod
    def calc_distance(p1, p2):
        calc = lambda x1, x2: pow(x1 - x2, 2)

        return sqrt(sum(map(calc, p1.dimensions, p2.dimensions)))
