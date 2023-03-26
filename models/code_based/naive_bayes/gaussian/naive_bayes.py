from math import sqrt, exp, pi

data_noise = 10 ** (-5)


class NaiveBayesClassifier:
    def __init__(self, space):
        
        self.class_mean = {}
        self.class_variance = {}
        self.class_prior = {}
        self.space = space
        self.calculate_class_mean_variance_prior()

    def calculate_class_mean_variance_prior(self):
        
        self.class_mean.clear()
        self.class_variance.clear()
        self.class_prior.clear()

        for classification in self.space.classifications:
            class_points = [point.dimensions for point in self.space.points if point.classifier == classification]

            class_mean, class_variance = self.calculate_mean_variance(class_points)

            self.class_mean[classification] = class_mean
            self.class_variance[classification] = class_variance
            self.class_prior[classification] = len(class_points)

    def classify(self, point):
       
        best_classification = None
        best_probability = 0

        for classification in self.space.classifications:
            cumulative_probability = self.class_prior[classification] / self.space.total_points()

            for dimension, (mean, variance) in enumerate(
                    zip(self.class_mean[classification], self.class_variance[classification])):
                probability = 1 / sqrt(2 * pi * (variance + data_noise)) * exp(
                    -((point.dimensions[dimension] - mean) ** 2) / (2 * (variance + data_noise)))
                cumulative_probability *= probability

            if cumulative_probability > best_probability:
                best_probability = cumulative_probability
                best_classification = classification

        return best_classification

    @staticmethod
    def calculate_mean_variance(data_points):
        
        mean = sum(data_points) / len(data_points)
        variance = sum(map(lambda x: pow(x - mean, 2), data_points)) / (len(data_points) - 1)
        return mean, variance
