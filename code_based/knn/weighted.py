from math import sqrt

data_noise = 10 ** (-5)


class KNNClassifier:
    def __init__(self, space):
        
        self.space = space

    def classify(self, query_point, k: int, weighted: bool) -> str:
        
        self.sort_points_by_distance(query_point)
        k_nearest_points = self.space.points[:k]

        votes = {}
        for point in k_nearest_points:
            distance = self.calculate_distance(query_point, point)
            weight = (1 if not weighted else max(1, 1 / (distance + data_noise)))

            if point.classifier in votes:
                votes[point.classifier] += weight
            else:
                votes[point.classifier] = weight

        return max(votes, key=votes.get)

    def sort_points_by_distance(self, query_point):
        
        self.space.points.sort(key=lambda p: self.calculate_distance(query_point, p))

    @staticmethod
    def calculate_distance(point1, point2) -> float:
        
        calc = lambda x1, x2: pow(x1 - x2, 2)
        return sqrt(sum(map(calc, point1.dimensions, point2.dimensions)))
