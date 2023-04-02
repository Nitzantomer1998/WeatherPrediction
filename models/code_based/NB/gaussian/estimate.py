# Library to calculate square root, pi, exp
from math import sqrt, exp, pi

# The amount of data_noise to add to the variance when using weighted voting
data_noise = 10 ** (-5)


class NaiveBayesClassifier:
    def __init__(self, space):
        """
        Initializes a NaiveBayesClassifier object with the given space.

        Args:
            space (Space): The space object containing the points to be classified.
        """
        self.class_mean = {}
        self.class_variance = {}
        self.class_prior = {}
        self.space = space
        self.calculate_class_mean_variance_prior()

    def calculate_class_mean_variance_prior(self):
        """
        Calculates the mean, variance and prior probabilities for each classification in the space.
        """
        # Initialize variables
        self.class_mean.clear()
        self.class_variance.clear()
        self.class_prior.clear()

        # Calculate mean and variance for each classification
        for classification in self.space.classifications:
            # Get data points that belong to the current classification
            class_points = [point.dimensions for point in self.space.points if point.classifier == classification]

            # Calculate the mean and variance of the current classification
            class_mean, class_variance = self.calculate_mean_variance(class_points)

            # Store the mean, variance, and prior probabilities of the current classification
            self.class_mean[classification] = class_mean
            self.class_variance[classification] = class_variance
            self.class_prior[classification] = len(class_points)

    def classify(self, point):
        """
        Classifies the given point using the Naive Bayes algorithm.

        Args:
            point (Point): The point to be classified.

        Returns:
            str: The classification label for the given point.
        """
        best_classification = None
        best_probability = 0

        # Calculate the probability of the point belonging to each classification
        for classification in self.space.classifications:
            cumulative_probability = self.class_prior[classification] / self.space.total_points()

            # Calculate the probability of each dimension of the point belonging to the current classification
            for dimension, (mean, variance) in enumerate(
                    zip(self.class_mean[classification], self.class_variance[classification])):
                probability = 1 / sqrt(2 * pi * (variance + data_noise)) * exp(
                    -((point.dimensions[dimension] - mean) ** 2) / (2 * (variance + data_noise)))
                cumulative_probability *= probability

            # Update the best classification if the probability of the current classification is higher than the best probability
            if cumulative_probability > best_probability:
                best_probability = cumulative_probability
                best_classification = classification

        return best_classification

    @staticmethod
    def calculate_mean_variance(data_points):
        """
        Calculates the mean and variance of the given data points.

        Args:
            data_points (list of tuple): The list of data points.

        Returns:
            tuple: A tuple of two values, the mean and variance of the given data points.
        """
        mean = sum(data_points) / len(data_points)
        variance = sum(map(lambda x: pow(x - mean, 2), data_points)) / (len(data_points) - 1)
        return mean, variance
