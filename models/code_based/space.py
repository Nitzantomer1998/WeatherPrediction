class Space:
    def __init__(self) -> None:
        self.points = []
        self.classifications = set()

    def add_point(self, point) -> None:
        self.points.append(point)
        self.classifications.add(point.classifier)

    def clear_space(self) -> None:
        self.points.clear()
        self.classifications.clear()
